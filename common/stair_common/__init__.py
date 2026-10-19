"""STAIR's shared pipeline pieces: frozen splits, the standard scores, AgentTTS-unit cost, the results layout."""

import tomllib
from pathlib import Path

MODELS_TOML = Path(__file__).resolve().parent / "models.toml"


def load_models() -> dict:
    """The pinned model menus: load_models()["A"]["answerer"]["Llama-3.1-70B"] -> {repo, revision, params_b, dtype}."""
    return tomllib.loads(MODELS_TOML.read_text(encoding="utf-8"))
