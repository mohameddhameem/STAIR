#!/usr/bin/env python3
"""
gpab_run.py -- run GPAgentBench-2K consultations: one agent model (weak or strong) against the local patient simulator.
DRAFT v1.0 2026-10-31 for Jonathan's line-by-line review. NOT to be launched before approval.

Pipeline per case (registration §2-§4):
  loop until terminal or 10 rounds:
    prompt = their C.1 prompt rebuilt for this turn (gpab_env.Episode.build_prompt)
    reply  = AGENT (vLLM OpenAI-compatible server), temperature 0, max_tokens 2048
    parse  -> one <action type=...> block
      malformed / masked (D1): ONE re-prompt = same prompt + notice line; second failure consumes the round, episode continues
    env.step(): ask -> SIMULATOR reply (our prompt, temperature 0); body_exam/test -> deterministic lookup; diagnose/refer/treat
  record -> <out>/episodes_<agent_tag>.jsonl   (full transcript, actions, token usage per call, dollars)
Replay cache: a case already present in the output file is skipped (one execution per (case, agent); registration §3.4).
Costs: prompt/completion token counts come from the server's `usage` (the served model's own tokenizer); FLOPs units are
computed in gpab_score.py with stair_common.cost.units -- this file only records the counts.

USAGE (cluster, inside the sbatch; two vLLM servers already up):
  python3 gpab_run.py --data <jsonl dir> --cases splits/gpagentbench/pilot60.txt --agent-tag weak \
      --agent-model Qwen/Qwen2.5-7B-Instruct --agent-url http://localhost:8001/v1 \
      --sim-model Qwen/Qwen2.5-7B-Instruct   --sim-url   http://localhost:8001/v1 --out <run dir>
Secrets: none. The servers are local; no API keys are read or written.
"""
import argparse, glob, hashlib, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpab_env as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PROMPT = open(os.path.join(HERE, "prompts", "patient_simulator_v1.txt"), encoding="utf-8").read()
SIM_RULES = json.load(open(os.path.join(HERE, "prompts", "patient_simulator_v1_persona_rules.json"), encoding="utf-8"))
D1_NOTICE = "\n\nYour previous response was not a valid action for this turn. Produce exactly one allowed action."


def chat(url, model, prompt, max_tokens, temperature=0.0, timeout=600):
    """One chat completion against a vLLM OpenAI-compatible endpoint. Returns (text, usage_dict)."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.load(r)
            return j["choices"][0]["message"]["content"], j.get("usage", {})
        except Exception as e:  # transient server errors: retry twice, then fail loudly
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def patient_record_text(patient):
    return "\n".join(f"- {k}: {v if v not in (None, '') else 'Not reported'}" for k, v in patient.items())


def simulator_prompt(case, doctor_utterance):
    p = case["persona"]
    lang = str(p.get("language_proficiency", "B")).upper()
    return SIM_PROMPT.format(
        patient_record=patient_record_text(case["patient"]),
        personality=p.get("personality", "neutral"),
        personality_rule=SIM_RULES["personality"].get(p.get("personality", "neutral"), SIM_RULES["personality"]["neutral"]),
        language_rule=SIM_RULES["language_proficiency"].get(lang, SIM_RULES["language_proficiency"]["B"]),
        recall_rule=SIM_RULES["recall_level"].get(p.get("recall_level", "high"), SIM_RULES["recall_level"]["high"]),
        disorientation_rule=SIM_RULES["disorientation"].get(p.get("disorientation", "normal"), SIM_RULES["disorientation"]["normal"]),
        max_words=SIM_RULES["max_words"].get(lang, 60),
        doctor_utterance=doctor_utterance)


def load_cases(data_dir, case_list):
    want = [l.strip() for l in open(case_list) if l.strip()]
    rows = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for k, v in list(r.items()):
                if isinstance(v, str) and v[:1] in "{[":
                    try:
                        r[k] = json.loads(v)
                    except Exception:
                        pass
            if r["case_id"] in want:
                rows[r["case_id"]] = r
    missing = [c for c in want if c not in rows]
    assert not missing, f"cases not found in data: {missing[:5]}"
    return [rows[c] for c in want]


def run_episode(case, a):
    ep = E.Episode(case)
    calls = []                                   # every model call: who, prompt_tokens, completion_tokens
    while not ep.done():
        prompt = ep.build_prompt()
        reply, usage = chat(a.agent_url, a.agent_model, prompt, a.max_tokens)
        calls.append({"who": "agent", "round": ep.round + 1, **{k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}})
        parsed = E.parse_action(reply)
        ok = parsed is not None and ep.allowed_now(parsed[0])
        if not ok:                               # D1: one re-prompt with the notice
            reply2, usage2 = chat(a.agent_url, a.agent_model, prompt + D1_NOTICE, a.max_tokens)
            calls.append({"who": "agent_reprompt", "round": ep.round + 1, **{k: usage2.get(k) for k in ("prompt_tokens", "completion_tokens")}})
            parsed = E.parse_action(reply2)
            ok = parsed is not None and ep.allowed_now(parsed[0])
            if not ok:                           # second failure: the round is consumed, nothing happens
                ep.round += 1
                ep.actions.append({"round": ep.round, "type": "invalid", "fields": {}, "body": (reply2 or "")[:500]})
                ep.transcript.append({"role": "System", "text": "[invalid action; round consumed]"})
                continue
        atype, fields, body = parsed
        obs, need_patient = ep.step(atype, fields, body)
        if need_patient:
            sp = simulator_prompt(case, body)
            answer, su = chat(a.sim_url, a.sim_model, sp, a.sim_max_tokens)
            calls.append({"who": "simulator", "round": ep.round, **{k: su.get(k) for k in ("prompt_tokens", "completion_tokens")}})
            ep.add_patient_reply(answer.strip())
    rec = ep.record()
    rec["calls"] = calls
    rec["agent_tag"] = a.agent_tag
    rec["agent_model"] = a.agent_model
    rec["sim_model"] = a.sim_model
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--cases", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--agent-tag", required=True, choices=["weak", "strong"])
    ap.add_argument("--agent-model", required=True); ap.add_argument("--agent-url", required=True)
    ap.add_argument("--sim-model", required=True); ap.add_argument("--sim-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--sim-max-tokens", type=int, default=256)
    ap.add_argument("--repeat-tag", default=None, help="write to a separate file (churn measurement), e.g. rep2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"episodes_{a.agent_tag}{'_' + a.repeat_tag if a.repeat_tag else ''}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["case_id"] for l in open(out) if l.strip()}
    cases = load_cases(a.data, a.cases)
    man = {"version": "gpab_run v1.0 2026-10-31", "agent": a.agent_tag, "agent_model": a.agent_model, "sim_model": a.sim_model,
           "max_tokens": a.max_tokens, "sim_max_tokens": a.sim_max_tokens, "temperature": 0.0, "max_rounds": E.MAX_ROUNDS,
           "cases_file": a.cases, "cases_sha256": hashlib.sha256(open(a.cases, "rb").read()).hexdigest(), "n_cases": len(cases),
           "agent_prompt_sha256": hashlib.sha256(open(E.PROMPT_PATH, "rb").read()).hexdigest(),
           "sim_prompt_sha256": hashlib.sha256(SIM_PROMPT.encode()).hexdigest(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(man, open(os.path.join(a.out, f"manifest_{a.agent_tag}.json"), "w"), indent=1)
    print(json.dumps(man, indent=1), flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            if case["case_id"] in done:
                continue
            t0 = time.time()
            rec = run_episode(case, a)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{i+1}/{len(cases)}] {case['case_id']} {rec['terminal']:8s} rounds {rec['rounds']:2d} ${rec['dollars']:4d} {time.time()-t0:5.0f}s", flush=True)
    print("DONE", out)


if __name__ == "__main__":
    main()
