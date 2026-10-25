"""Jev novelty E9: milliseconds per routing decision, on the same real inputs for every router.

The decision timed is menu A's answer-stage call (experiment 10) on the first --n slr-test questions: the question,
the weak retriever's references and the weak answer, each router reading them its own way:

    nano-jev judge        the saved golden-set qpa judge (runs/slr-judge-models), exp10's typed qpa item
    MS MARCO MiniLM-L12   the same 33M architecture and item (a fine-tuned one times the same)
    Qwen2.5-1.5B router   Jonathan's INSTR_QA1 prompt (exp10.his_router_calls), one forward pass to the next-token
                          logits, as his routers decide (their LoRA adds nothing once merged)
    Strands Decider 2B / E4B   exp10's typed qpa item through DeciderJudge
    NLI xsmall            cross-encoder/nli-deberta-v3-xsmall: the references entail "the answer is X"?

Batch 1: p50 / p95 over the questions after a warm-up; batch 32: ms per decision. Every timing includes tokenisation.
On a CPU, torch uses --threads threads. Writes <out>/<device>.json.

    latency --device cuda --out runs/e9-latency        # GPU job
    latency --device cpu --threads 8 --out runs/e9-latency   # CPU job (ncpus=8)
"""

import argparse
import json
import logging
import time
import tomllib
from pathlib import Path

import numpy as np

from stair import exp10_slr, provenance

log = logging.getLogger(__name__)

JUDGE = "runs/slr-judge-models/judges/qpa-large-helps-nanojev-A/seed0/model"
MSMARCO = "cross-encoder/ms-marco-MiniLM-L12-v2"
DECIDERS = {"Strands Decider 2B": "StrandsAgents/strands-decider-2B-hobson-v21",
            "Strands Decider E4B": "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"}
NLI = "cross-encoder/nli-deberta-v3-xsmall"
CLAIM = 'The answer to "{question}" is {answer}.'


def timed(fn, batches: list, warmup: int = 5) -> list[float]:
    for b in batches[:warmup]:
        fn(b)
    out = []
    for b in batches:
        start = time.perf_counter()
        fn(b)
        out.append(1000 * (time.perf_counter() - start))
    return out


def summary(single: list[float], batched: list[float], batch: int) -> dict:
    return {"batch1_p50_ms": float(np.percentile(single, 50)), "batch1_p95_ms": float(np.percentile(single, 95)),
            f"batch{batch}_ms_per_decision": float(np.mean(batched) / batch), "n": len(single)}


def main(argv: list[str] | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    from stair.judge import make_judge

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n", type=int, default=200, help="questions timed at batch 1 (and in batches of --batch)")
    parser.add_argument("--n-large", type=int, default=50, help="questions for the LLM-sized routers on a CPU")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--plan", type=Path, default=Path("configs/exp10-slr.toml"))
    parser.add_argument("--out", type=Path, default=Path("runs/e9-latency"))
    parser.add_argument("--skip", action="append", default=[], help="router names to leave out")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    plan = tomllib.loads(args.plan.read_text(encoding="utf-8"))
    df = exp10_slr.load(plan)
    test = df[df["split"] == "test"].reset_index(drop=True).head(args.n)
    items = exp10_slr.items(test, "qpa")["qa_W"]
    slr_plan = tomllib.loads(Path(plan["slr_plan"]).read_text(encoding="utf-8"))
    system, calls = exp10_slr.his_router_calls(test, exp10_slr.his_prompts(Path(slr_plan["out"]) / exp10_slr.HIS_PROMPTS))
    report = {"device": args.device, "threads": args.threads if args.device == "cpu" else None,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "routers": {}}

    def run(name: str, fn, inputs: list, n: int) -> None:
        if name in args.skip:
            return
        single = timed(fn, [[x] for x in inputs[:n]])
        batched = timed(fn, [inputs[i: i + args.batch] for i in range(0, min(n, len(inputs)) - args.batch + 1, args.batch)] or
                        [inputs[: args.batch]], warmup=1)
        report["routers"][name] = summary(single, batched, args.batch)
        log.info(f"{name:24s} {report['routers'][name]}")

    large_n = args.n if args.device == "cuda" else args.n_large
    for name, path in (("nano-jev judge (33M)", JUDGE), ("MS MARCO MiniLM-L12 (33M)", MSMARCO)):
        j = make_judge(path, args.device, 512, args.batch)
        run(name, lambda b, j=j: j.probabilities(b), items, args.n)
        del j
    for name, path in DECIDERS.items():
        if name in args.skip:
            continue
        if args.device == "cpu":  # the deciders' Qwen3.5 / Gemma torsos run flash-linear-attention's Triton kernels
            report["routers"][name] = {"error": "needs a GPU: its Triton kernels have no CPU driver (job 43240)"}
            continue
        try:
            j = make_judge(path, args.device, 1024, args.batch)
            run(name, lambda b, j=j: j.probabilities(b), items, large_n)
            del j
        except Exception as e:  # one router failing must not lose the others' timings
            report["routers"][name] = {"error": f"{type(e).__name__}: {e}"}
            log.warning(f"{name}: {e}")
        torch.cuda.empty_cache() if args.device == "cuda" else None

    if "Qwen2.5-1.5B router" not in args.skip:
        tok = AutoTokenizer.from_pretrained(exp10_slr.HIS_ROUTER, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(exp10_slr.HIS_ROUTER, dtype=torch.bfloat16 if args.device == "cuda"
                                                  else torch.float32).to(args.device).eval()
        texts = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                         tokenize=False, add_generation_prompt=True) for u in calls["qa_W"]]

        def his(batch: list[str]) -> None:
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                lm(**enc).logits[:, -1].float().softmax(-1)

        run("Qwen2.5-1.5B router", his, texts, large_n)
        del lm

    if "NLI xsmall (71M)" not in args.skip:
        tok = AutoTokenizer.from_pretrained(NLI)
        nli = AutoModelForSequenceClassification.from_pretrained(NLI).to(args.device).eval()
        pairs = [(sel, CLAIM.format(question=q, answer=a)) for q, sel, a in zip(test["question"], test["sel_W"], test["answer_WW"])]

        def entail(batch: list[tuple[str, str]]) -> None:
            enc = tok([p for p, _ in batch], [c for _, c in batch], truncation="only_first", max_length=512,
                      padding=True, return_tensors="pt").to(args.device)
            with torch.no_grad():
                nli(**enc).logits.softmax(-1)

        run("NLI xsmall (71M)", entail, pairs, args.n)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.device}.json").write_text(json.dumps(report, indent=1))
    provenance.record(args.out, f"latency-{args.device}", **provenance.environment())


if __name__ == "__main__":
    main()
