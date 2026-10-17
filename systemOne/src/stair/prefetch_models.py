"""Download every model a config needs into the Hugging Face cache.

Run this where there is internet (e.g. a cluster login node) before running on GPU nodes that
have none; then set HF_HUB_OFFLINE=1 there. Set HF_HOME to a disk with room (~20 GB for 1.5B + 7B).
"""

import argparse
import logging
import tomllib
from pathlib import Path

from stair.provenance import hub_revision

log = logging.getLogger(__name__)

# Llama repos ship a second copy of the weights in original/ (141 GB for the 70B) that transformers never loads.
IGNORE = ["original/*"]


def models_of(cfg: dict) -> list[tuple[str, str | None]]:
    """(repo id, revision) for the scorer and each expert that loads from the Hub."""
    out = []
    path = cfg.get("scorer", {}).get("path")
    if path and not Path(path).is_dir():
        repo, _, rev = path.partition("@")
        out.append((repo, rev or None))
    for expert in cfg["experts"]:
        if "model" in expert and not Path(expert["model"]).is_dir():
            out.append((expert["model"], expert.get("revision")))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, action="append", required=True, help="repeatable")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from huggingface_hub import snapshot_download

    for config in args.config:
        for repo, revision in models_of(tomllib.loads(config.read_text())):
            log.info(f"{repo}@{revision or 'main'} ...")
            snapshot_download(repo, revision=revision, ignore_patterns=IGNORE)
            log.info(f"  cached at commit {hub_revision(repo, revision)}")


if __name__ == "__main__":
    main()
