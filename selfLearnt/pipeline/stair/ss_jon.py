"""AL-02 replication on Jonathan's own inputs (his reply of 30 Oct): the always-strong ANSWER step re-run on our stack.

Inputs, exactly his:
  - AgentTTS's HotpotQA pool file tts_test.json (7,405 questions, 100-passage pools in their own order; team Shared files,
    sha256 f65d581370fb27b9...), at paths.tts_test or <project_dir>/hotpotqa/tts_test.json
  - his stored strong-retriever selections, branch jonathan: jonathan/runs/config_b/retr_strong.jsonl (key "idx")
  - his answering prompt, lab_package/run_labels4.py qa_prompt(): AgentTTS's answer prompt plus "Answer with ONLY the exact
    answer phrase - no explanation, no full sentence." (spec section 6, corrected 28 Oct)
Then one greedy pass of Llama 3.1 70B (the pinned AWQ copy, vLLM), 128 new tokens, system "You are a helpful assistant.",
and a question-by-question comparison with his stored answers (jonathan/runs/config_b/qa_strong.jsonl, key "idx|strong").
Same answers => the serving stacks match and the 4-bit format question is settled; a gap points at quantization.

  python run.py --config configs/default.yaml --stage ss_jon --part check     (CPU: pools rebuilt in his order?)
  python run.py --config configs/default.yaml --stage ss_jon --part answer    (GPU: Llama 70B)
  python run.py --config configs/default.yaml --stage ss_jon --part compare   (CPU)
"""
import hashlib
import json
import os
import random
import re
import subprocess

from .common import append_jsonl, log, read_jsonl, write_json
from .config import serve_entry, variant
from .engines import free_gpu, make_engine
from .progress import mark_done

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
POOL_K = 100
QA_SYS = "You are a helpful assistant."
QA_OUT = 128


# ---------------- his code (lab_package/run_labels4.py), kept byte for byte ----------------
def title_of(c):
    return c.split(':', 1)[0]


def stable_seed(s):
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


def trim_pool(inst, k=POOL_K, shuffle=False):
    cands = inst['ctxs_candidate']
    golds = inst.get('supporting_titles', [])
    pool = list(cands[:k])
    have = set(title_of(c) for c in pool)
    for t in golds:
        if t not in have:
            g = next((c for c in cands if title_of(c) == t), None)
            if g:
                for j in range(k - 1, -1, -1):
                    if title_of(pool[j]) not in golds:
                        pool[j] = g
                        break
                have.add(t)
    if shuffle:
        rnd = random.Random(42 + stable_seed(inst['_id']) % 100000)
        rnd.shuffle(pool)
    return pool


def qa_prompt(inst, retrieved):
    return (f"Your task is to answer the query using the provided references. "
            f"If the references are insufficient, supplement your response with your own knowledge. "
            f"Answer with ONLY the exact answer phrase - no explanation, no full sentence.\n"
            f"Query: {inst['question']}\n{retrieved}")


def parse_selection(txt, n_refs):
    part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
    nums = [int(x) for x in re.findall(r'\[(\d+)\]', part)]
    return sorted(set(n for n in nums if 1 <= n <= n_refs))


def build_block(inst, sel):
    if not sel:
        return f"Query: {inst['question']}\nSelected references: (none)"
    parts = [f"[{n}] {inst['ctxs_candidate'][n-1]}" for n in sel]
    return f"Query: {inst['question']}\nSelected references: " + " ".join(parts)


# ---------------- inputs ----------------
def _git_jsonl(path):
    out = subprocess.run(["git", "-C", ROOT, "show", f"origin/jonathan:{path}"], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"cannot read {path} from branch jonathan ({out.stderr.strip()}): run `git fetch origin jonathan` first")
    return [json.loads(x) for x in out.stdout.splitlines() if x.strip()]


def _tts(cfg):
    p = (os.environ.get("STAIR_TTS_TEST") or cfg["paths"].get("tts_test")
         or os.path.join(cfg["paths"]["project_dir"], "hotpotqa", "tts_test.json"))
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found: copy AgentTTS's tts_test.json there (team Shared files, Dataset/HotpotQA_AgentTTS)")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    log(f"tts_test.json sha256 {h[:16]}... ({'matches' if h.startswith('f65d581370fb27b9') else 'DIFFERS FROM'} Jonathan's f65d581370fb27b9...)")
    return json.load(open(p, encoding="utf-8"))


def _gold(inst):
    g = inst.get("answers", inst.get("answer"))
    return g if isinstance(g, list) else [g]


def _qid(inst):
    return inst.get("_id") or inst.get("id")


def wdir(cfg):
    d = os.path.join(cfg["run_dir"], "ss_jon")
    os.makedirs(d, exist_ok=True)
    return d


def _setup(cfg, split):
    """[(idx, inst with his pool, his selection)] for the split's questions, and which pool order matches his echoes."""
    from stair_common.splits import load_split
    tts = _tts(cfg)
    retr = {r["k"]: r["v"] for r in _git_jsonl("jonathan/runs/config_b/retr_strong.jsonl")}
    pos = {_qid(x): i for i, x in enumerate(tts)}
    ids = load_split("hotpotqa", split)
    missing = [q for q in ids if q not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} {split} questions are not in tts_test.json (e.g. {missing[:3]})")
    # his July store does not record whether the pool was shuffled: take the order in which the passages he selected carry
    # the titles his retriever echoed ("[5] Ed Wood: ...")
    score = {}
    for shuffle in (False, True):
        hit = tot = 0
        for q in ids[:300]:
            i = pos[q]
            inst = dict(tts[i])
            pool = trim_pool(inst, shuffle=shuffle)
            txt = retr.get(str(i), "")
            part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
            for n, title in re.findall(r'\[(\d+)\]\s*([^:\[\]]{2,80}):', part):
                n = int(n)
                if 1 <= n <= len(pool):
                    tot += 1
                    hit += title_of(pool[n - 1]).strip() == title.strip()
        score[shuffle] = hit / max(1, tot)
    shuffle = score[True] > score[False]
    log(f"pool order: echoed titles match {100 * score[False]:.1f}% as given, {100 * score[True]:.1f}% shuffled -> "
        f"{'shuffled' if shuffle else 'as given'}")
    rows = []
    for q in ids:
        i = pos[q]
        inst = dict(tts[i])
        inst['ctxs_candidate'] = trim_pool(inst, shuffle=shuffle)
        sel = parse_selection(retr[str(i)], len(inst['ctxs_candidate']))
        rows.append((i, q, inst, sel))
    return rows, max(score.values())


def run(cfg, part, split="slr-test"):
    out = os.path.join(wdir(cfg), f"{split}_ans_l70.jsonl")
    if part == "check":
        rows, match = _setup(cfg, split)
        log(f"{split}: {len(rows)} questions, {sum(1 for r in rows if r[3])} with a non-empty selection; "
            f"title match {100 * match:.1f}% ({'OK' if match > 0.95 else 'TOO LOW: the pools differ from his - stop here'})")
        return
    if part == "answer":
        rows, match = _setup(cfg, split)
        if match <= 0.95:
            raise SystemExit(f"pools rebuilt from tts_test.json match his selections only {100 * match:.1f}%: not his inputs")
        have = {r["qid"] for r in read_jsonl(out)}
        todo = [r for r in rows if r[1] not in have]
        log(f"{split}: {len(todo)} of {len(rows)} answers to generate (Llama 3.1 70B, greedy, his prompt)")
        if todo:
            m = variant(cfg, "l70")
            eng = make_engine(cfg, serve_entry(m))
            chunk = cfg["gpu"]["chunk"]
            for c in range(0, len(todo), chunk):
                part_r = todo[c:c + chunk]
                prompts = [eng.encode(QA_SYS, qa_prompt(inst, build_block(inst, sel))) for _, _, inst, sel in part_r]
                outs = eng.generate(prompts, 1, 0.0, QA_OUT, [0] * len(part_r),
                                    [{"kind": "qa_free", "gold": _gold(inst)[0], "has_gold": True} for _, _, inst, _ in part_r])
                append_jsonl(out, [{"qid": q, "idx": i, "prompt_tokens": len(p), "text": o[0][0], "out_tokens": o[0][1]}
                                   for (i, q, _, _), p, o in zip(part_r, prompts, outs)])
            free_gpu(eng)
        mark_done(cfg, "ss_jon", "answer")
        return
    # compare: ours vs his stored answers, question by question, on the same ruler
    from stair_common import score
    tts = _tts(cfg)
    his = {r["k"]: r["v"] for r in _git_jsonl("jonathan/runs/config_b/qa_strong.jsonl")}
    ours = read_jsonl(out)
    if not ours:
        raise SystemExit("no answers yet: run --part answer first")
    n = same_text = a = b = only_ours = only_his = 0
    for r in ours:
        inst = tts[r["idx"]]
        g = _gold(inst)
        h = his.get(f"{r['idx']}|strong", "")
        lo, lh = score.lenient(r["text"], g), score.lenient(h, g)
        n += 1
        a += lo
        b += lh
        same_text += r["text"].strip() == h.strip()
        only_ours += lo and not lh
        only_his += lh and not lo
    gap = 100 * (a - b) / n
    agree = 1 - (only_ours + only_his) / n
    verdict = ("REPLICATED: the serving stacks match" if abs(gap) <= 0.5 and agree >= 0.95 else
               "CLOSE: small serving difference" if abs(gap) <= 2 else "NOT REPLICATED: serving differs (quantization?)")
    msg = (f"{split}: {n} questions; ours {100 * a / n:.2f} vs his {100 * b / n:.2f} lenient ({gap:+.2f} pt); identical answer text "
           f"{100 * same_text / n:.1f}%; same right/wrong {100 * agree:.1f}% (ours only {only_ours}, his only {only_his}) -> {verdict}")
    log(msg)
    write_json(os.path.join(wdir(cfg), f"{split}_compare.json"),
               {"n": n, "ours": a / n, "his": b / n, "identical_text": same_text / n, "same_verdict": agree,
                "ours_only": only_ours, "his_only": only_his, "verdict": verdict})
