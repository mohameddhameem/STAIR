#!/usr/bin/env python3
# =============================================================================
# tworouter_trainer.py — VERSION: v0.4.2 2026-11-28 (DRAFT for Jonathan's line-by-line review; NOT yet run)
# "Two-router stage-level routing on MedAgentBench: Router 1 at every information lookup (W/S),
#  Router 2 at every decision stage (W/S/H)."
#
# LINEAGE: training6_trainer.py v1.2 (priced reward, GRPO A1, fresh LoRA, mean-baseline advantage, entropy BETA,
# kill baseline, overwrite guard). Everything not listed under DELTA is carried over unchanged.
#
# DELTA over Training 6:
#   * STAGE REPLAY instead of one end-of-attempt view. The stored weak episode (WeakPilot7B_run1) is sliced
#     into stages: round t with action GET  -> information lookup  -> Router 1 is called after the reply;
#                  round t with action POST/FINISH -> decision stage -> Router 2 is called BEFORE it is committed;
#                  episode ends with no POST/FINISH (8-round cap or 'agent context limit')
#                                           -> AMENDMENT (--cap-decision, default on): Router 2 is called once
#                                              with "has prepared: no decision (stopped without an order or a final answer / context limit reached)".
#     Pre-registration check (runs/mab/tworouter_check_28Nov2026.txt, script d06edad745af, commit 8eea7cb):
#     weak lookups 1439, order decisions 60 (types 3, 8), final answers 131, 'agent context limit' 8 (type 9);
#     types 5/6/7: 30 x 8 lookups, no decision stage. Without the amendment Router 2 never fires on them.
#   * TWO ROUTERS = two LoRA adapters ("router1", "router2") on ONE frozen base (Qwen2.5-1.5B-Instruct), same LoRA
#     shape as the SFT router, both fresh, both trained by the same GRPO objective on the trajectory reward.
#   * TRAJECTORY: at lookup t Router 1 samples W (carry on) or S (strong restarts: outcome q_s, cost c_w,t + c_s);
#     after the last lookup Router 2 samples W (decision goes through: outcome q_w, cost c_w), S (as above with
#     c_w,t = c_w) or H (only tau_i >= TAU_GATE: outcome Q_H = 1, AI cost c_w, one hand-off unit) — as Training 6.
#     Decision stages after an H never happen on the stored data (H is only offered on task5/task9, whose weak
#     episodes have no POST/FINISH), so H ends the instance; asserted in build_rows.
#   * REWARD: one instance reward R' for the whole trajectory, the Training 6 form (LAMBDA, hinge, priced failure on
#     the tier, outcome-free H), credited to every sampled router step on that trajectory (REINFORCE over the
#     trajectory, group mean baseline over G trajectories of the same instance). Stage inputs do not depend on
#     earlier router verdicts, so the per-stage log-probabilities are computed ONCE per instance per epoch and
#     the G trajectories are sampled from them (<= 9 forwards per instance, not 9 x G).
#   * DECISION RULE: greedy argmax at every stage (K = 1), as Training 6. --theta-h is reserved (not used in v0.1).
#   * PROMPTS: Router 1 / Router 2 system and user messages as on master-deck slide 51 (28 Nov). The two sentences
#     kept verbatim from the Training 5/6 prompt are asserted. Reply cut REPLY_CUT = 600 (registered rule).
#   * REMOVED: Training 1's table (training1_table.json) is no longer the router input; it is still read for
#     split, q_w, q_s, c_s and the whole-episode c_w, and c_w,t is recomputed with the same tokenizer and the
#     same (no-separator) context accounting so that c_w,T == table c_w (asserted, tolerance 0.006: the table stores round(cw, 2)).
#
# THREE CHECKS (Jonathan's convention), all before any registered run:
#   1. hashes   : --dry-run prints sha256 of this script and of every input file; the manifest records them.
#   2. inputs   : --dry-run (no GPU) rebuilds the stage table and asserts the pre-check totals above, then prints the
#                 first Router 1 and Router 2 messages of one instance per task type for reading; --smoke then runs
#                 6 optimizer steps on 20/30 rows into <outdir>_smoke.
#   3. guard    : overwrite guard on --outdir; prompt self-check; single-token option letters; H never offered on a
#                 below-gate row; no decision stage after H (data assertion).
#
# Usage (lab, slr-env):
#   python3 tworouter_trainer.py --dry-run
#   python3 tworouter_trainer.py --price operating --smoke
#   python3 tworouter_trainer.py --price operating --explore-epochs 10 --baseline <always-escalate val mean-R'>
# =============================================================================
import os, sys, json, time, random, argparse, hashlib, collections
import torch

VERSION = "tworouter_trainer v0.4.2 2026-11-28 (mean_pW per router in eval; per-adapter update-norm check after training; Training-6b-policy anchor = kill bar; c_w,T tolerance 0.006 — table stores 2 dp; escalation-rule sentence removed from SYS2_HIGH, kept on file as ablation E5b; {actor} slot in transcript lines; Router 2 system wording per Jonathan 22:15; two LoRA routers on one base; stage replay of WeakPilot7B_run1; trajectory GRPO A1; cap-decision amendment default on)"

MAB     = "/mnt/hdd/jonathan/medagentbench"
WEAK    = f"{MAB}/outputs/WeakPilot7B_run1/qwen2.5-7b-local/medagentbench-std/overall.json"
STRONG  = f"{MAB}/outputs/MedAgentBenchv1/qwen2.5-72b-local/medagentbench-std/overall.json"
DATA    = f"{MAB}/data/medagentbench/test_data_v2.json"
TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"        # split, qw, qs, cs, cw (whole episode)
TAU     = "/mnt/hdd/jonathan/runs/mab/tau_scores_v2.json"
PRICES  = "/mnt/hdd/jonathan/runs/mab/tier_prices_v1_computed.json"
TAU_GATE = 0.80
LAMBDA_T = 1.0
Q_H = 1.0
MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
WEAK_TOK = "Qwen/Qwen2.5-7B-Instruct"            # cost accounting tokenizer, as training1_build_table.py
LORA_SHAPE_FROM = "/mnt/hdd/jonathan/runs/config_b/sft_router5_ret/best_adapter"
WEAK_NAME, STRONG_NAME = "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"
LR, ACCUM, SEED, BETA, LAMBDA = 1e-5, 8, 17, 0.03, 0.10
EVAL_EVERY = 10
REPLY_CUT = 600          # registered rule (tworouter_check: 61/61 weak, 321/321 strong key-bearing replies pass)
DECISION_CUT = 600       # same cut on the prepared order / final answer text
ROUND_CAP = 8            # MedAgentBench agent round limit

# ---- pre-registration check totals (runs/mab/tworouter_check_28Nov2026.txt) asserted in --dry-run -------------
PRECHECK = {"instances": 300, "information lookup": 1439, "order decision": 60, "final answer": 131, "other": 8}

# ---- the router prompts (master deck slide 51, 28 Nov; for Jonathan's review) ---------------------------------
OBJ  = ("Your objective: the task should be completed correctly while spending as little compute as possible "
        "- correctness first, cost second.")
BASED = "Based on the information shown below, choose one option."
SYS1 = ("You route a medical record-system task one step at a time. The weak model {weak} is working on it; the "
        "transcript so far is shown below, ending with the information lookup it has just completed. Decide whether "
        "the weak model carries on, or the task is escalated to {strong}, which redoes the task from the start. "
        + OBJ + " " + BASED).format(weak=WEAK_NAME, strong=STRONG_NAME)
SYS2_HEAD = ("You route a medical record-system task one step at a time. The weak model {weak} is working on it; the "
             "transcript so far is shown below, ending with the decision it has prepared - an order to write to the "
             "record, or its final answer. The decision has not been carried out yet. Decide which one happens: the "
             "weak model's decision goes through, ").format(weak=WEAK_NAME)
SYS2_LOW  = (SYS2_HEAD + "or the task is escalated to {strong}, which redoes the task from the start. ".format(strong=STRONG_NAME)
             + OBJ + " " + BASED)
SYS2_HIGH = (SYS2_HEAD + ("the task is escalated to {strong}, which redoes the task from the start, or the decision is "
             "escalated to a doctor. This decision is high-risk: a wrong order or answer that reaches the patient costs "
             "far more than a doctor's time, and a doctor's time costs more than either model. ").format(strong=STRONG_NAME)
             + OBJ + " " + BASED)
TAIL1  = "\nThis step is: information lookup\nOptions:\nW) The weak model carries on\nS) Escalate the task to the strong model\nAnswer with a single letter: W or S."
TAIL2_LOW  = "\nOptions:\nW) The weak model's decision goes through\nS) Escalate the task to the strong model\nAnswer with a single letter: W or S."
TAIL2_HIGH = "\nOptions:\nW) The weak model's decision goes through\nS) Escalate the task to the strong model\nH) Escalate the decision to a doctor\nAnswer with a single letter: W, S or H."


def _prompt_selfcheck():
    for s in (SYS1, SYS2_LOW, SYS2_HIGH):
        assert OBJ in s and BASED in s, "objective / based-on sentence drifted"
        for banned in ("Routing principles", "only when needed", "only when", "Weigh that", "(1)", "(2)", "(3)", "cheap)", "expensive)"):
            assert banned not in s, f"banned (principle-flavored) phrase: {banned}"
    assert "doctor" not in SYS2_LOW and "doctor" in SYS2_HIGH
    assert TAIL2_HIGH.count("\n") == TAIL2_LOW.count("\n") + 1, "H adds exactly one option line"
    assert "H) Escalate the decision to a doctor" in TAIL2_HIGH and "H)" not in TAIL2_LOW and "H)" not in TAIL1
    print("[tworouter prompt] self-check OK", flush=True)
_prompt_selfcheck()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hinge(x): return max(0.0, x - 1.0)


def stage_type(action):
    a = action.strip()
    if a.startswith("FINISH"): return "final answer"
    if a.startswith("POST"):   return "order decision"
    if a.startswith("GET"):    return "information lookup"
    return "other"


def load_episodes(p):
    return {r["index"]: r for r in json.load(open(p))["custom"]["raw_results"]}


def walk(rec):
    """(round_no, action, reply) from the stored history: agent turn, then the record system's reply (or None)."""
    h = rec["history"]; out = []; k = 0
    for i, m in enumerate(h):
        if m["role"] != "agent": continue
        k += 1
        reply = h[i + 1]["content"] if i + 1 < len(h) and h[i + 1]["role"] == "user" else None
        out.append((k, m["content"], reply))
    return out


def weak_costs_by_round(rec, tok):
    """Cumulative weak cost (AgentTTS units) after each agent round, with training1_build_table.py's accounting:
    prompt tokens = tokens of all prior turns concatenated without separators; generated = tokens of the agent turn."""
    from stair_common import cost as C
    n_in = n_out = 0; ctx = ""; cum = []
    for m in rec["history"]:
        if m["role"] == "agent":
            n_in += len(tok.encode(ctx)); n_out += len(tok.encode(m["content"]))
            cum.append(C.units(7.0, n_in, n_out))
        ctx += m["content"]
    return cum


def fmt_round(k, action, reply, actor="weak model"):
    """One transcript line. `actor` is who produced the round: 'weak model' / 'strong model' / 'doctor'. In Run A
    (restart semantics) every routed round is the weak model's; the slot exists so Run B (per-stage, one escalation)
    keeps the same prompt format."""
    rep = (reply if reply is not None else "")[:REPLY_CUT]
    return f"Round {k} — {actor}: {action.strip()} → record system: {rep}"


def build_messages(task_text, steps, cap_decision):
    """Router calls for one weak episode.
    Returns a list of stages, each {"router": 1|2, "round": k, "stype": ..., "user": str, "decision": action|None}.
    Router 1 after each lookup round (transcript through that round); Router 2 at each POST/FINISH round
    (lookups so far + 'has prepared: ...'); if the episode ends with no decision and cap_decision is on, one final
    Router 2 stage with 'no decision (round limit reached)'."""
    stages, lines = [], []
    has_decision = False
    for k, action, reply in steps:
        st = stage_type(action)
        if st == "information lookup":
            lines.append(fmt_round(k, action, reply))
            stages.append({"router": 1, "round": k, "stype": st, "decision": None,
                           "user": f"{task_text}\nTranscript so far:\n" + "\n".join(lines) + TAIL1})
        elif st in ("order decision", "final answer"):
            has_decision = True
            prepared = f"Round {k} — weak model has prepared: {action.strip()[:DECISION_CUT]}"
            stages.append({"router": 2, "round": k, "stype": st, "decision": action.strip()[:DECISION_CUT],
                           "user": f"{task_text}\nTranscript so far:\n" + "\n".join(lines + [prepared]) + f"\nThis step is: {st}"})
            lines.append(prepared)   # a later FINISH after a POST sees the order it follows
        else:                        # 'agent context limit' etc.: the episode stops here
            break
    if not has_decision and cap_decision:
        k = len(steps)
        why = "context limit reached" if stage_type(steps[-1][1]) == "other" else "stopped without an order or a final answer"
        prepared = f"After round {k} — weak model has prepared: no decision ({why})"
        stages.append({"router": 2, "round": k, "stype": "no decision", "decision": None,
                       "user": f"{task_text}\nTranscript so far:\n" + "\n".join(lines + [prepared]) + "\nThis step is: no decision"})
    return stages


def load_prices(price_setting):
    P = json.load(open(PRICES))
    assert P["tau_gate"] == TAU_GATE, (P["tau_gate"], TAU_GATE)
    rho = {task: v["settings"][price_setting]["rho"] for task, v in P["task_types"].items()}
    C_H = {task: v["C_H"] for task, v in P["task_types"].items()}
    C_F = {task: v["settings"][price_setting]["C_F"] for task, v in P["task_types"].items()}
    print(f"[tworouter prices] setting={price_setting}  rho={rho}  C_H={C_H}  C_F={C_F}", flush=True)
    return rho, C_H, C_F


def build_rows(rho, C_H, C_F, cap_decision, splits=("train", "val"), tok=None):
    """One row per instance with its stage list and everything the replay needs. Correctness and tau are consumed
    into the reward; neither appears in any router input (asserted: no 'completedCorrect' in any user message)."""
    from transformers import AutoTokenizer
    tok = tok or AutoTokenizer.from_pretrained(WEAK_TOK)
    W = load_episodes(WEAK)
    table = {t["idx"]: t for t in json.load(open(TABLE))}
    data = {x["id"]: x for x in json.load(open(DATA))}
    taus = {s["tid"]: s["tau"] for s in json.load(open(TAU))["scores"]}
    rows = collections.defaultdict(list); totals = collections.Counter(); n_inst = 0
    for idx in sorted(W):
        tid = f"task{idx//30+1}_{idx%30+1}"; t = table[idx]
        assert t["tid"] == tid
        n_inst += 1
        steps = walk(W[idx])
        for _, action, _ in steps: totals[stage_type(action)] += 1
        if t["split"] not in splits: continue
        d = data[tid]
        task_text = d["instruction"] + (("\nContext: " + d["context"]) if d.get("context", "").strip() else "")
        cum = weak_costs_by_round(W[idx], tok)
        assert abs(cum[-1] - t["cw"]) < 0.006, f"{tid}: c_w,T {cum[-1]:.4f} != table c_w {t['cw']:.4f}"   # table stores round(cw, 2)
        high = taus[tid] >= TAU_GATE
        task = tid.split("_")[0]
        stages = build_messages(task_text, steps, cap_decision)
        for s in stages:
            assert "completedCorrect" not in s["user"] and "completedIncorrect" not in s["user"]
            s["cw_t"] = cum[s["round"] - 1]            # weak cost paid up to and including this round
            if s["router"] == 2:
                s["sys"] = SYS2_HIGH if high else SYS2_LOW
                s["user"] = s["user"] + (TAIL2_HIGH if high else TAIL2_LOW)
            else:
                s["sys"] = SYS1
        dec = [s for s in stages if s["router"] == 2]
        if high:
            assert task in rho, f"no price for {task}"
            assert len(dec) <= 1, f"{tid}: high-risk instance with {len(dec)} decision stages (H would not be terminal)"
        r = {"idx": idx, "tid": tid, "task": task, "split": t["split"], "high": high, "tau": taus[tid],
             "qw": t["qw"], "qs": t["qs"], "cw": t["cw"], "cs": t["cs"], "stages": stages,
             "rho": rho[task] if high else 0.0, "C_H": C_H.get(task) if high else None, "C_F": C_F.get(task) if high else None,
             "both_fail": (t["qw"] == 0 and t["qs"] == 0)}
        rows[t["split"]].append(r)
    totals["instances"] = n_inst
    for k, v in PRECHECK.items():
        assert totals[k] == v, f"pre-check mismatch {k}: got {totals[k]}, registered {v}"
    print(f"[tworouter] pre-check totals reproduced: {dict(totals)}", flush=True)
    for sp, rs in rows.items():
        n1 = sum(1 for r in rs for s in r["stages"] if s["router"] == 1)
        n2 = sum(1 for r in rs for s in r["stages"] if s["router"] == 2)
        ncap = sum(1 for r in rs for s in r["stages"] if s["stype"] == "no decision")
        hi = [r for r in rs if r["high"]]
        print(f"[tworouter] {sp}: n={len(rs)}  router1 calls={n1}  router2 calls={n2} (of which no-decision={ncap})  "
              f"high-tier={len(hi)}  both-fail on high tier={sum(r['both_fail'] for r in hi)}", flush=True)
    return rows


# ---- trajectory outcome: the Training 6 reward on the delivered path ---------------------------------------------
def outcome(r, path):
    """path = list of (stage_index, action). Returns (q, ai_cost, action_class, pen_rho) where action_class in
    {'W','S','H'} is the terminal action and the reward is assembled by reward()."""
    for si, a in path:
        s = r["stages"][si]
        if a == "S":
            return r["qs"], s["cw_t"] + r["cs"], "S"
        if a == "H":
            assert r["high"] and s["router"] == 2
            return Q_H, s["cw_t"], "H"
    # W throughout: the weak episode runs to its end
    return r["qw"], r["cw"], "W"


def reward(r, q, ai_cost, term, rho_for_grad=None):
    x = ai_cost / r["cs"]
    if term == "H":
        return Q_H - LAMBDA * x - LAMBDA_T * 1.0
    rp = q - LAMBDA * x - hinge(x)
    if r["high"] and q == 0:
        rho = r["rho"] if rho_for_grad is None else min(r["rho"], rho_for_grad)
        rp -= LAMBDA_T * rho
    return rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="default: runs/mab/tworouter_<price>[_nocap]")
    ap.add_argument("--price", choices=["lower", "central", "operating"], default="operating")
    ap.add_argument("--no-cap-decision", action="store_true", dest="no_cap_decision",
                    help="turn the amendment OFF: no Router 2 call when the weak episode ends without POST/FINISH")
    ap.add_argument("--rho-cap", type=float, default=None, dest="rho_cap",
                    help="cap on rho_i in the TRAINING penalty only; evaluation keeps full rho_i. REGISTERED VALUE 15 (as Training 6b)")
    ap.add_argument("--G", type=int, default=8, help="trajectories per instance")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--explore-epochs", type=int, default=0, dest="explore_epochs",
                    help="epochs during which every stage samples uniformly over its allowed actions")
    ap.add_argument("--baseline", type=float, default=None, help="kill-criterion bar: always-escalate val mean-R'")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="checks 1-2: hashes, stage table, pre-check totals, example messages; no GPU; writes nothing")
    a = ap.parse_args()
    cap_decision = not a.no_cap_decision

    inputs = {"script": os.path.abspath(__file__), "weak": WEAK, "strong": STRONG, "data": DATA, "table": TABLE,
              "tau": TAU, "prices": PRICES}
    hashes = {k: sha256(p) for k, p in inputs.items()}
    print("[tworouter hashes] " + json.dumps(hashes, indent=1), flush=True)

    rho, C_H, C_F = load_prices(a.price)
    rows = build_rows(rho, C_H, C_F, cap_decision)

    if a.dry_run:
        seen = set()
        for sp in ("train", "val"):
            for r in rows[sp]:
                if r["task"] in seen: continue
                seen.add(r["task"])
                s1 = next((s for s in r["stages"] if s["router"] == 1), None)
                s2 = next((s for s in r["stages"] if s["router"] == 2), None)
                print(f"\n########## {r['tid']}  split={sp}  high={r['high']}  tau={r['tau']}  stages={len(r['stages'])}  "
                      f"qw={r['qw']} qs={r['qs']} cw={r['cw']:.2f} cs={r['cs']:.2f}")
                if s1: print(f"--- ROUTER 1, round {s1['round']} ---\n[system] {s1['sys']}\n[user]\n{s1['user']}")
                if s2: print(f"--- ROUTER 2, round {s2['round']} ({s2['stype']}) ---\n[system] {s2['sys']}\n[user]\n{s2['user']}")
        # anchors: always-W, always-S, oracle (greedy over the three terminal outcomes), per split
        for sp in ("train", "val"):
            rs = rows[sp]
            mW = sum(reward(r, r["qw"], r["cw"], "W") for r in rs) / len(rs)
            mS = sum(reward(r, r["qs"], r["cw"] + r["cs"], "S") for r in rs) / len(rs)
            def s_at(r, k):      # escalate at the k-th router call (or at the last call if the episode is shorter)
                st = r["stages"][min(k, len(r["stages"])) - 1]; return reward(r, r["qs"], st["cw_t"] + r["cs"], "S")
            m1 = sum(s_at(r, 1) for r in rs) / len(rs); m2 = sum(s_at(r, 2) for r in rs) / len(rs)
            def t6b(r):          # Training 6b's learned policy: S at call 1 below the gate; on the tier, H at the (only) decision stage
                if not r["high"]: return s_at(r, 1)
                st = [x for x in r["stages"] if x["router"] == 2][-1]; return reward(r, Q_H, st["cw_t"], "H")
            m6 = sum(t6b(r) for r in rs) / len(rs)
            print(f"[anchors] {sp}: always-W mean R' {mW:+.4f} | always-S after the whole weak attempt (Training 6 bar) {mS:+.4f} | "
                  f"always-S at call 1 {m1:+.4f} | at call 2 (Training 3's rule) {m2:+.4f} | "
                  f"S-below-gate + H-on-tier (Training 6b policy) {m6:+.4f}  <- --baseline = this last value (val)")
        print("\n[dry-run] OK — nothing written", flush=True)
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

    if a.outdir is None:
        a.outdir = f"/mnt/hdd/jonathan/runs/mab/tworouter_{a.price}" + ("" if cap_decision else "_nocap") \
                   + (f"_cap{a.rho_cap:g}" if a.rho_cap is not None else "")
    if a.smoke:
        a.outdir = a.outdir.rstrip("/") + "_smoke"
    if os.path.exists(a.outdir) and not a.smoke:
        raise SystemExit(f"REFUSING: {a.outdir} exists — a registered run dir is never re-entered; pass a new --outdir")
    os.makedirs(a.outdir, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    if a.smoke:
        rows["train"] = rows["train"][:20]; rows["val"] = rows["val"][:30]
        a.G, eval_every, max_steps = 4, 2, 6
    else:
        eval_every, max_steps = EVAL_EVERY, None
    print(f"{VERSION}  price={a.price}  cap_decision={cap_decision}  G={a.G}  train={len(rows['train'])}  "
          f"val={len(rows['val'])}  epochs={a.epochs}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    def one_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"letter {s!r} is not a single token: {ids}"
        return ids[0]
    idW, idS, idH = one_id("W"), one_id("S"), one_id("H")

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    cfg = PeftConfig.from_pretrained(LORA_SHAPE_FROM); cfg.inference_mode = False
    lcfg = {k: v for k, v in cfg.to_dict().items() if k in LoraConfig.__dataclass_fields__}
    model = get_peft_model(base, LoraConfig(**lcfg), adapter_name="router1")
    model.add_adapter("router2", LoraConfig(**lcfg))
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in params: p.requires_grad_(True)
    print(f"two fresh routers: trainable params = {sum(p.numel() for p in params):,} "
          f"(= 2 x the HotpotQA fresh-run count)", flush=True)
    opt = torch.optim.AdamW(params, lr=LR)
    ACT = ["W", "S", "H"]

    def use(router):
        model.set_adapter(f"router{router}")
        for p in params: p.requires_grad_(True)     # set_adapter freezes the inactive adapter; keep both trainable

    def logprobs(stage, high):
        """log-probabilities over the allowed actions at this stage: [W,S] for Router 1 and low-risk Router 2,
        [W,S,H] for high-risk Router 2."""
        use(stage["router"])
        msgs = [{"role": "system", "content": stage["sys"]}, {"role": "user", "content": stage["user"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids): ids = ids["input_ids"]
        out = model(input_ids=ids.to("cuda"), logits_to_keep=1).logits[0, -1]
        allow_h = (stage["router"] == 2 and high)
        logits = [out[idW], out[idS]] + ([out[idH]] if allow_h else [])
        return torch.log_softmax(torch.stack(logits), 0)

    def stage_lps(r):
        return [logprobs(s, r["high"]) for s in r["stages"]]

    def greedy_path(r, lps):
        path = []
        for si, lp in enumerate(lps):
            c = ACT[int(lp.argmax())]; path.append((si, c))
            if c != "W": break
        return path

    def evaluate(vrows):
        model.eval()
        tot_r = tot_acc = tot_cost = 0.0; n = len(vrows)
        term_count = collections.Counter(); r1_S = r1_calls = 0
        pw = {1: [], 2: []}                     # mean P(W) per router: moves long before the greedy argmax flips
        hi = collections.Counter()
        with torch.no_grad():
            for r in vrows:
                lps = stage_lps(r); path = greedy_path(r, lps)
                for st, lp in zip(r["stages"], lps): pw[st["router"]].append(float(lp[0].exp()))
                q, cost, term = outcome(r, path)
                tot_r += reward(r, q, cost, term); tot_acc += q; tot_cost += cost
                term_count[term] += 1
                r1_calls += sum(1 for si, _ in path if r["stages"][si]["router"] == 1)
                r1_S += sum(1 for si, c in path if r["stages"][si]["router"] == 1 and c == "S")
                if r["high"]:
                    hi["n"] += 1; hi[term] += 1; hi["acc"] += q; hi["both_fail"] += r["both_fail"]
                    hi["H_on_both_fail"] += (term == "H" and r["both_fail"])
                    if term != "H": hi["fail_no_human"] += (1 - q); hi["failure_cost_sgd"] += r["C_F"] * (1 - q)
                    else: hi["human_cost_sgd"] += r["C_H"]
        model.train()
        H = hi["H"]
        return {"val_meanR_delivered": round(tot_r / n, 4), "val_acc": round(tot_acc / n, 4),
                "val_mean_ai_cost": round(tot_cost / n, 1),
                "mean_pW_router1": (round(sum(pw[1]) / len(pw[1]), 4) if pw[1] else None),
                "mean_pW_router2": (round(sum(pw[2]) / len(pw[2]), 4) if pw[2] else None),
                "terminal": dict(term_count), "router1_calls": r1_calls, "router1_escalations": r1_S,
                "high_tier": {"n": hi["n"], "deliver": hi["W"], "escalate": hi["S"], "to_doctor": H,
                              "acc": (round(hi["acc"] / hi["n"], 4) if hi["n"] else None),
                              "H_precision": (round(hi["H_on_both_fail"] / H, 4) if H else None),
                              "H_recall": (round(hi["H_on_both_fail"] / hi["both_fail"], 4) if hi["both_fail"] else None),
                              "fail_with_no_human": hi["fail_no_human"],
                              "human_cost_sgd": round(hi["human_cost_sgd"], 2), "failure_cost_sgd": round(hi["failure_cost_sgd"], 2)}}

    ev0 = evaluate(rows["val"])
    print(json.dumps({"step": 0, "FRESH_START_EVAL": True, **ev0, "baseline": a.baseline}), flush=True)

    snap = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}   # for the update check
    def delta_norms():
        d = {"router1": 0.0, "router2": 0.0}
        for n, p in model.named_parameters():
            if "lora_" in n:
                d["router1" if ".router1." in n else "router2"] += float((p.detach() - snap[n]).norm() ** 2)
        return {k: round(v ** 0.5, 6) for k, v in d.items()}
    curve = open(os.path.join(a.outdir, "curve.jsonl"), "a")
    best_obj, step, t0, stop = -1e9, 0, time.time(), False
    order = rows["train"][:]
    for ep in range(a.epochs):
        if stop: break
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            lps = stage_lps(r)                                    # one forward per stage, with grad
            probs = [lp.exp().detach().tolist() for lp in lps]
            if ep < a.explore_epochs:
                probs = [[1.0 / len(p)] * len(p) for p in probs]
            draws = []
            for _ in range(a.G):                                  # G trajectories sampled from the stage policies
                path = []
                for si, p in enumerate(probs):
                    c = random.choices(ACT[:len(p)], weights=p)[0]; path.append((si, c))
                    if c != "W": break
                q, cost, term = outcome(r, path)
                draws.append((path, reward(r, q, cost, term, rho_for_grad=a.rho_cap)))
            rewards = [d[1] for d in draws]; mu = sum(rewards) / a.G
            pg = 0.0
            for (path, _), A in zip(draws, [x - mu for x in rewards]):   # A1: mean baseline, no /std
                if A == 0.0: continue
                for si, c in path:
                    pg = pg + A * lps[si][ACT.index(c)]
            ent = sum(-(lp.exp() * lp).sum() for lp in lps) / len(lps)
            loss = (-(pg / a.G) - BETA * ent) / ACCUM
            if not torch.isfinite(loss):
                print(f"skipped non-finite loss (epoch {ep}, row {k})", flush=True); continue
            loss.backward()
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % eval_every == 0:
                    ev = evaluate(rows["val"])
                    line = {"step": step, "epoch": ep, **ev, "baseline": a.baseline,
                            "beats_baseline": (None if a.baseline is None else ev["val_meanR_delivered"] > a.baseline),
                            "min": round((time.time() - t0) / 60, 1)}
                    print(json.dumps(line), flush=True); curve.write(json.dumps(line) + "\n"); curve.flush()
                    if ev["val_meanR_delivered"] > best_obj:
                        best_obj = ev["val_meanR_delivered"]
                        model.save_pretrained(os.path.join(a.outdir, "best_adapter"))   # saves router1/ and router2/
                if max_steps and step >= max_steps:
                    stop = True; break
    model.save_pretrained(os.path.join(a.outdir, "last_adapter"))
    dn = delta_norms()
    print(json.dumps({"UPDATE_CHECK": True, "steps": step, "param_change_norm": dn}), flush=True)
    assert dn["router1"] > 0 and dn["router2"] > 0, f"an adapter received no update: {dn}"

    # ---- final: reload BEST, score val per stage ----------------------------------------------------------------
    del model; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    model = PeftModel.from_pretrained(base, os.path.join(a.outdir, "best_adapter", "router1"), adapter_name="router1")
    model.load_adapter(os.path.join(a.outdir, "best_adapter", "router2"), adapter_name="router2")
    model.eval(); params = []
    final = evaluate(rows["val"])
    with torch.no_grad(), open(os.path.join(a.outdir, "val_stage_scores.jsonl"), "w") as fv:
        for r in rows["val"]:
            for si, s in enumerate(r["stages"]):
                pr = logprobs(s, r["high"]).exp().tolist()
                fv.write(json.dumps({"idx": r["idx"], "tid": r["tid"], "stage": si, "router": s["router"], "round": s["round"],
                                     "stype": s["stype"], "high": r["high"], "pW": round(pr[0], 6), "pS": round(pr[1], 6),
                                     "pH": (round(pr[2], 6) if len(pr) == 3 else None)}) + "\n")
    manifest = {"stage": "tworouter", "version": VERSION, "inputs": inputs, "sha256": hashes,
                "price_setting": a.price, "rho_by_task": rho, "C_H_by_task": C_H, "C_F_by_task": C_F,
                "lambda": LAMBDA, "lambda_t": LAMBDA_T, "Q_H": Q_H, "tau_gate": TAU_GATE, "rho_cap_in_gradient": a.rho_cap,
                "cap_decision_amendment": cap_decision, "reply_cut": REPLY_CUT, "decision_cut": DECISION_CUT,
                "decision_rule": "greedy argmax at every stage (K=1)", "routers": "two LoRA adapters on one base, both fresh",
                "reward": "Training 6 form on the trajectory outcome; credited to every sampled step; mean-baseline advantage; entropy BETA",
                "prompts": {"router1_sys": SYS1, "router2_sys_low": SYS2_LOW, "router2_sys_high": SYS2_HIGH,
                            "tail1": TAIL1, "tail2_low": TAIL2_LOW, "tail2_high": TAIL2_HIGH},
                "base_model": MODEL, "lora_shape_from": LORA_SHAPE_FROM,
                "lr": LR, "epochs": a.epochs, "explore_epochs": a.explore_epochs, "accum": ACCUM, "seed": a.seed,
                "beta": BETA, "G": a.G, "smoke": a.smoke, "kill_baseline": a.baseline,
                "final_fullval": final, "fresh_start_eval": ev0}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)


if __name__ == "__main__":
    main()
