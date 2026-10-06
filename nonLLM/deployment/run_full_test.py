"""Run the full test benchmark, render its report, and publish only its summary artifacts."""
import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-push", action="store_true", help="Save report for synchronization from another host")
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    root = Path("/ssd1/zmcheng/cs707")
    repo = root / "STAIR"
    run = root / "runs/full_pipeline_test_v1"
    script = Path(__file__).with_name("full_pipeline_eval.py")
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    subprocess.run([sys.executable, str(script), "--split", "test", "--gpu", str(args.gpu), "--out", str(run)],
                   env=env, check=True)
    result = json.loads((run / "results.json").read_text())
    assert len(result["config"]["ids"]) == 1481
    progress = json.loads((run / "status.json").read_text())
    assert progress["state"] == "complete" and progress["calls_done"] == 8886
    lines = ["# Full actual-model test benchmark", "",
             "All 1,481 frozen internal test questions. Actual 1.5B Base / 7B Base Q4_K_M retrieval and answering.",
             "2,962 retrieval calls and 5,924 QA calls are complete. There are no skipped test questions.",
             "Retrieval sees every candidate passage and selects up to four original passages with constrained IDs.",
             "Two synthetic retrieval demonstrations are fixed in the prompt; no gold answers or supporting annotations enter prompts.",
             "Every QA branch uses NEW evidence produced in this run. Existing classifiers remain frozen.", "",
             "| Method | Answer EM % | Answer F1 % | Diagnostic containment % | WW / WS / SW / SS |",
             "|---|---:|---:|---:|---|"]
    for name, metrics in result["policies"].items():
        picks = " / ".join(str(metrics["picks"].get(cell, 0)) for cell in ["WW", "WS", "SW", "SS"])
        lines.append(f"| {name} | {metrics['answer_em_pct']:.2f} | {metrics['answer_f1_pct']:.2f} | "
                     f"{metrics['diagnostic_containment_pct']:.2f} | {picks} |")
    lines += ["", f"Actual four-cell oracle answer EM: {result['oracle_em_pct']:.2f}%.", "",
              "EM and token F1 use the official HotpotQA answer-scoring functions. No supporting-fact or joint score is claimed.",
              "Containment is only a diagnostic, not the official metric or a certified reproduction of the original lenient scorer.",
              "Raw timings were collected with two concurrent requests and other GPU experiments. They are not isolated routed latency.",
              "The full grid was generated once, then each frozen policy selected its actual retrieval/answering path.",
              "This is a real-model grid evaluation, not nine separately timed online deployments.",
              "Prompts, quantization, selected evidence, and generation settings differ from the old stored-grid experiment.",
              "The full test result does not modify training labels, classifiers, or thresholds.", "",
              "Configuration, hashes, scores and timings: [results.json](results.json).", "",
              "Raw retrieval/QA outputs and resumable checkpoints are stored locally in `runs/full_pipeline_test_v1/calls.sqlite`.",
              "Every method's selected path is stored locally in `decisions.jsonl`.", ""]
    (run / "REPORT.md").write_text("\n".join(lines))
    target = repo / "nonLLM/results/full_pipeline_test_v1"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["REPORT.md", "results.json"]:
        shutil.copyfile(run / name, target / name)
    if args.no_push:
        (run / "publication.json").write_text(json.dumps({"state": "ready_for_sync", "branch": "nonLLM"})+"\n")
        print("Complete full-test report saved for synchronization.", flush=True)
        return
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "nonLLM":
        print("Report saved locally; checkout branch changed, so publication was skipped", flush=True)
        return
    subprocess.run(["git", "add", "nonLLM/results/full_pipeline_test_v1/REPORT.md",
                    "nonLLM/results/full_pipeline_test_v1/results.json"], cwd=repo, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", "nonLLM/results/full_pipeline_test_v1"], cwd=repo)
    if changed.returncode:
        subprocess.run(["git", "commit", "--only", "-m", "Report full actual-model routing test results", "--",
                        "nonLLM/results/full_pipeline_test_v1/REPORT.md", "nonLLM/results/full_pipeline_test_v1/results.json"],
                       cwd=repo, check=True)
    subprocess.run(["git", "push", "origin", "nonLLM"], cwd=repo, check=True)
    (run / "publication.json").write_text(json.dumps({"state": "published", "branch": "nonLLM"})+"\n")
    print("Complete full-test report saved and synchronized.", flush=True)


if __name__ == "__main__":
    main()
