"""CPU-only local monitor and result synchronization for the serverC GPU run."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time

HOST = "SMU_C_4A5000"
ROOT = Path("/ssd1/zmcheng/cs707")
RUN = ROOT / "runs/full_pipeline_test_v1"


def remote(command):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, command],
                          capture_output=True, text=True, timeout=30)


def main():
    while True:
        reply = remote(f"cat {RUN}/status.json")
        if reply.returncode:
            print("Waiting for serverC status:", reply.stderr.strip(), flush=True)
            time.sleep(30)
            continue
        progress = json.loads(reply.stdout)
        progress["execution_host"] = "gpuserver1 (10.193.104.142)"
        progress["execution_gpu"] = 2
        progress["local_gpu_used"] = False
        temp = RUN / "status.json.sync.tmp"
        temp.write_text(json.dumps(progress, indent=2)+"\n")
        temp.replace(RUN / "status.json")
        print(json.dumps(progress), flush=True)
        if progress["state"] == "failed":
            print("Remote run stopped; no automatic restart. User notification required.", flush=True)
            return
        if progress["state"] == "complete":
            ready = remote(f"test -f {RUN}/REPORT.md")
            if ready.returncode:
                time.sleep(5)
                continue
            break
        time.sleep(30)
    names = ["REPORT.md", "results.json", "calls.sqlite", "decisions.jsonl", "publication.json"]
    with tempfile.TemporaryDirectory(prefix="serverC-sync-", dir=RUN) as staging:
        for name in names:
            subprocess.run(["scp", "-o", "BatchMode=yes", f"{HOST}:{RUN}/{name}", str(Path(staging)/name)], check=True)
        results = json.loads((Path(staging)/"results.json").read_text())
        assert len(results["config"]["ids"]) == 1481
        db = sqlite3.connect(Path(staging)/"calls.sqlite")
        assert db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 8886
        db.close()
        # Keep the old 136-call snapshot separately; replace only the stopped local run.
        for name in names:
            shutil.copyfile(Path(staging)/name, RUN/name)
    repo = ROOT / "STAIR"
    target = repo / "nonLLM/results/full_pipeline_test_v1"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["REPORT.md", "results.json"]:
        shutil.copyfile(RUN/name, target/name)
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "nonLLM":
        print("Results copied locally; branch changed, so GitHub publication was skipped.", flush=True)
        return
    paths = ["nonLLM/results/full_pipeline_test_v1/REPORT.md", "nonLLM/results/full_pipeline_test_v1/results.json"]
    subprocess.run(["git", "add", *paths], cwd=repo, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo)
    if changed.returncode:
        subprocess.run(["git", "commit", "--only", "-m", "Report full actual-model routing test on serverC", "--", *paths], cwd=repo, check=True)
    subprocess.run(["git", "push", "origin", "nonLLM"], cwd=repo, check=True)
    (RUN/"publication.json").write_text(json.dumps({"state": "published", "source_host": HOST})+"\n")
    print("serverC results validated, copied and published.", flush=True)


if __name__ == "__main__":
    main()
