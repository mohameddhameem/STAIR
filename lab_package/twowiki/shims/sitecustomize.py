"""
sitecustomize.py -- TOKEN-COUNT INSTRUMENTATION for AgentTTS's unchanged archon_main.py (12 Nov 2026, DRAFT for Jonathan's review).

WHY. The 31 Oct / 1 Nov reproduction runs logged only the fused response and the retrieval ids; to MEASURE their pipeline's
cost (strong retrieval -> 3B answering x50 -> 3B fuser) instead of estimating it, every model call must record how many
tokens went in and came out. Python imports `sitecustomize` automatically at start-up when it is on PYTHONPATH, so this file
is picked up by the run script's existing `PYTHONPATH=code_others:shims` with NO change to their code.

WHAT IT DOES. Wraps `transformers.GenerationMixin.generate` (the one entry point their script uses for the 72B-AWQ retriever,
the 3B generator and the 3B fuser). Per call it appends one JSON line to $STAIR_SIDECAR (same file as the other shims):
  {"kind": "gen", "model": <config._name_or_path>, "n_seq_in": batch, "n_in": sum of prompt tokens over the batch,
   "n_seq_out": returned sequences, "n_out": sum of generated tokens (prompt removed; pad tokens not counted), "t": time}
Prompts, sampling parameters, outputs and return values are untouched: the wrapper only reads shapes after the call.
Nothing is recorded unless STAIR_SIDECAR is set. If the wrap fails for any reason it prints a line and leaves generate as is
(the run is then NOT a cost measurement; the post-run check in the run script will say so).
"""
import json
import os
import time

_SIDECAR = os.environ.get("STAIR_SIDECAR")


def _install():
    if not _SIDECAR:
        return
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception as e:  # transformers not importable yet / not used
        print(f"[stair sitecustomize] transformers not patched: {e}", flush=True)
        return
    orig = GenerationMixin.generate

    def generate(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            if ids is None and "inputs" in kwargs:
                ids = kwargs["inputs"]
            seqs = out.sequences if hasattr(out, "sequences") else out
            n_seq_in, n_in = (int(ids.shape[0]), int(ids.shape[0] * ids.shape[1])) if ids is not None else (0, 0)
            gen = seqs[:, ids.shape[1]:] if ids is not None else seqs
            pad = getattr(getattr(self, "generation_config", None), "pad_token_id", None)
            if pad is None:
                pad = getattr(self.config, "pad_token_id", None)
            n_out = int((gen != pad).sum()) if pad is not None else int(gen.numel())
            rec = {"kind": "gen", "model": getattr(self.config, "_name_or_path", "?"), "n_seq_in": n_seq_in, "n_in": n_in,
                   "n_seq_out": int(seqs.shape[0]), "n_out": n_out, "t": time.time()}
            with open(_SIDECAR, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # never break their run over bookkeeping
            print(f"[stair sitecustomize] token record failed: {e}", flush=True)
        return out

    GenerationMixin.generate = generate
    print("[stair sitecustomize] transformers.generate instrumented for token counts ->", _SIDECAR, flush=True)


_install()
