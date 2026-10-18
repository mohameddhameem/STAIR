"""Check the experiment entries (experiments/<id>.md) against the template in experiments/README.md.

    python experiments/check.py              # every entry in experiments/
    python experiments/check.py AL-01.md     # or the files named

Each entry's YAML block must parse and have the template's keys (nested ones too), an `id` that matches its file
name, a `code.commit` that is a commit of this repository, at most 2 GPUs and a walltime, and the team's headline
metric and cost unit. No entry may contain an email address or anything that looks like an access token. Prints one
line per problem and exits 1 if there is any. Needs PyYAML. Examples (example-*.md) are checked for keys only.
"""

import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NOT_ENTRIES = {"README.md", "CLAUDE-INSTRUCTIONS.md"}
MAY_BE_NONE = {"reproduce", "adopt"}  # a string starting with "none" instead of the template's table / list
BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
TOKEN = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b")


def yaml_block(text: str):
    match = BLOCK.search(text)
    return yaml.safe_load(match.group(1)) if match else None


def key_problems(template, entry, where: str) -> list[str]:
    """Missing or extra keys of `entry` against `template`, recursing into tables and the items of lists."""
    if isinstance(template, dict):
        if not isinstance(entry, dict):
            name = where.rsplit(".", 1)[-1]
            ok = name in MAY_BE_NONE and isinstance(entry, str) and entry.lower().startswith("none")
            return [] if ok else [f"{where}: expected a table with keys {list(template)}"]
        out = [f"{where}.{k}: missing" for k in template if k not in entry]
        out += [f"{where}.{k}: not in the template" for k in entry if k not in template]
        return out + [p for k in template if k in entry for p in key_problems(template[k], entry[k], f"{where}.{k}")]
    if isinstance(template, list) and template and isinstance(template[0], dict):
        if not isinstance(entry, list):
            name = where.rsplit(".", 1)[-1]
            ok = name in MAY_BE_NONE and isinstance(entry, str) and entry.lower().startswith("none")
            return [] if ok else [f"{where}: expected a list of tables"]
        return [p for i, item in enumerate(entry) for p in key_problems(template[0], item, f"{where}[{i}]")]
    return []


def is_commit(sha: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=HERE, capture_output=True).returncode == 0


def check(path: Path, template: dict) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems = [f"looks like an email address: {m.group(0)}" for m in EMAIL.finditer(text)]
    problems += ["looks like an access token" for _ in TOKEN.finditer(text)]
    try:
        entry = yaml_block(text)
    except yaml.YAMLError as e:
        return problems + [f"the YAML block does not parse: {e}"]
    if not isinstance(entry, dict):
        return problems + ["no ```yaml block"]
    problems += key_problems(template, entry, "entry")
    if path.name.startswith("example-"):
        return problems
    if str(entry.get("id")) != path.stem:
        problems.append(f"id {entry.get('id')!r} does not match the file name {path.stem!r}")
    if entry.get("status") not in ("new", "rerun"):
        problems.append("status must be new or rerun")
    if entry.get("status") == "rerun" and not isinstance(entry.get("reproduce"), dict):
        problems.append("a rerun needs the reproduce table (old_results, source, differences)")
    sha = (entry.get("code") or {}).get("commit")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)):
        problems.append(f"code.commit must be a full 40-character SHA in quotes if it is all digits, got {sha!r}")
    elif not is_commit(sha):
        problems.append(f"code.commit {sha} is not a commit of this repository: push it first")
    compute = entry.get("compute") or {}
    if not (isinstance(compute.get("gpus"), int) and 0 <= compute["gpus"] <= 2):
        problems.append("compute.gpus must be 0, 1 or 2 (2 is the per-user limit)")
    if not isinstance(compute.get("walltime_h"), (int, float)) or compute["walltime_h"] <= 0:
        problems.append("compute.walltime_h must be a number of hours")
    metrics = entry.get("metrics") or {}
    if metrics.get("headline") != "lenient" or metrics.get("cost_unit") != "agenttts":
        problems.append("metrics: the headline is lenient and the cost unit agenttts (the team standard)")
    return problems


def main(argv: list[str]) -> int:
    template = yaml_block((HERE / "README.md").read_text(encoding="utf-8"))
    files = [HERE / a if not Path(a).is_absolute() and not Path(a).exists() else Path(a) for a in argv] or \
        sorted(p for p in HERE.glob("*.md") if p.name not in NOT_ENTRIES)
    bad = 0
    for path in files:
        problems = check(path, template)
        bad += bool(problems)
        print(f"{path.name}: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
        for p in problems:
            print(f"  - {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
