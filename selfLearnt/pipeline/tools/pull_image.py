"""Unpack a Docker Hub image into a folder without Docker (the SMU cluster has no Docker / Apptainer).
Used for MedAgentBench's FHIR server image (jyxsu6/medagentbench), so the server can run with plain Java inside a job.

  python tools/pull_image.py jyxsu6/medagentbench:latest ~/stair-data/medagentbench-image

Downloads the linux/amd64 manifest, its config and layers (each checked against its sha256), unpacks the layers in
order into <out>/rootfs (Docker's whiteout files delete what an earlier layer added), and prints what the image would
run: entrypoint, command, working directory, environment, ports, and the Java runtimes and .jar / database files in it."""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request

REG = "https://registry-1.docker.io"
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])


def get(url, token, accept=None, stream=False):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", **({"Accept": accept} if accept else {})})
    r = urllib.request.urlopen(req, timeout=120)
    return r if stream else r.read()


def _remove(p):
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p)
    else:
        os.remove(p)


def unpack(path, root):
    """One layer (gzip or plain tar) onto root; returns the number of entries written."""
    n = 0
    with tarfile.open(path, mode="r|*") as tf:
        for m in tf:
            name = m.name.lstrip("./")
            if not name or name.startswith("/") or ".." in name.split("/"):
                continue
            base, leaf = os.path.split(name)
            if leaf == ".wh..wh..opq":                   # opaque directory: drop what earlier layers put in it
                d = os.path.join(root, base)
                if os.path.isdir(d):
                    for x in os.listdir(d):
                        _remove(os.path.join(d, x))
                continue
            if leaf.startswith(".wh."):                  # whiteout: an earlier layer's file is deleted
                t = os.path.join(root, base, leaf[4:])
                if os.path.lexists(t):
                    _remove(t)
                continue
            if m.ischr() or m.isblk() or m.isfifo():
                continue
            dest = os.path.join(root, name)
            if os.path.lexists(dest) and not (m.isdir() and os.path.isdir(dest) and not os.path.islink(dest)):
                _remove(dest)
            if m.islnk():                                # hard link: copy the file already unpacked
                src = os.path.join(root, m.linkname.lstrip("./"))
                if os.path.isfile(src):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src, dest)
                    n += 1
                continue
            m.uid = m.gid = os.getuid()
            tf.extract(m, root, set_attrs=False)
            if m.isfile():
                os.chmod(dest, (m.mode | 0o600) & 0o777)
            elif m.isdir():
                os.chmod(dest, 0o755)
            n += 1
    return n


def fix_links(root):
    """Absolute links inside the image (e.g. Debian's Java: conf/security/java.security -> /etc/java-17-openjdk/...) point at
    the host's / outside a container; rewrite each to the same place inside the unpacked root. Returns how many."""
    n = 0
    for d, dirs, files in os.walk(root):
        for f in dirs + files:
            p = os.path.join(d, f)
            if os.path.islink(p):
                t = os.readlink(p)
                if t.startswith("/"):
                    os.remove(p)
                    os.symlink(os.path.relpath(os.path.join(root, t.lstrip("/")), d), p)
                    n += 1
    return n


def main(ref, out):
    repo, _, tag = ref.partition(":")
    tag = tag or "latest"
    tok = json.load(urllib.request.urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull", timeout=60))["token"]
    man = json.loads(get(f"{REG}/v2/{repo}/manifests/{tag}", tok, ACCEPT))
    if "manifests" in man:                               # multi-platform index: take linux/amd64
        d = next(m["digest"] for m in man["manifests"] if m.get("platform", {}).get("architecture") == "amd64"
                 and m.get("platform", {}).get("os") == "linux")
        man = json.loads(get(f"{REG}/v2/{repo}/manifests/{d}", tok, ACCEPT))
    cfg = json.loads(get(f"{REG}/v2/{repo}/blobs/{man['config']['digest']}", tok))
    root = os.path.join(out, "rootfs")
    os.makedirs(root, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "image_config.json"), "w"), indent=1)
    print(f"{ref}: {len(man['layers'])} layers, {sum(l.get('size', 0) for l in man['layers']) / 1e9:.2f} GB compressed")
    for i, layer in enumerate(man["layers"]):
        dig = layer["digest"]
        path = os.path.join(out, dig.replace(":", "_") + ".tar")
        if not os.path.exists(path):
            h = hashlib.sha256()
            with get(f"{REG}/v2/{repo}/blobs/{dig}", tok, stream=True) as r, open(path + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    f.write(b)
            if "sha256:" + h.hexdigest() != dig:
                sys.exit(f"layer {dig}: checksum mismatch")
            os.rename(path + ".part", path)
        n = unpack(path, root)
        os.remove(path)
        print(f"  layer {i + 1}/{len(man['layers'])}: {n} entries")
    print(f"absolute links rewritten inside the image: {fix_links(root)}")
    c = cfg.get("config", {})
    print("\nentrypoint:", c.get("Entrypoint"), "\ncmd:", c.get("Cmd"), "\nworkdir:", c.get("WorkingDir"),
          "\nports:", list((c.get("ExposedPorts") or {}).keys()), "\nenv:", c.get("Env"))
    found = {"java": [], "jar/war": [], "database": []}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            if f == "java" and d.endswith("/bin"):
                found["java"].append(p)
            elif f.endswith((".jar", ".war")) and os.path.getsize(p) > 5e6:
                found["jar/war"].append(f"{p} ({os.path.getsize(p) / 1e6:.0f} MB)")
            elif f.endswith((".mv.db", ".h2.db", ".db", ".sqlite")) and os.path.getsize(p) > 1e6:
                found["database"].append(f"{p} ({os.path.getsize(p) / 1e6:.0f} MB)")
    for k, v in found.items():
        print(f"{k}: " + ("; ".join(v[:6]) if v else "none"))


if __name__ == "__main__":
    if sys.argv[1] == "--fix-links":        # repair an image unpacked before this step existed
        print(f"absolute links rewritten: {fix_links(os.path.join(os.path.expanduser(sys.argv[2]), 'rootfs'))}")
    else:
        main(sys.argv[1], os.path.expanduser(sys.argv[2]))
