"""
Stage-Level Routing — FULL HotpotQA label run (lab GPU version)
Ports the Colab top-30 pilot (vC design) to a plain script: no Colab, no Drive.

Design is LOCKED to match the pilots exactly (comparability):
  top-30 BM25 pools + gold re-injection + seeded shuffle
  one-shot retrieval (greedy), raw-passage QA input (vC), official EM downstream
  weak = Qwen2.5-1.5B fp16 (always) · strong = Qwen2.5-7B, dtype selectable via --dtype

Guard stack: expandable-segments allocator, SDPA, 4096-token cap, per-item OOM -> save + fail-fast,
append-only JSONL checkpoints (resume = just rerun the same command).

CHANGES vs the 28 Jul version (all documented in CHANGELOG at the bottom of this docstring):
  1. Pool shuffle is now genuinely deterministic (hashlib, not Python's salted hash()).
  2. --dtype {4bit,fp16} replaces hand-editing the quantization_config lines.
  3. run_manifest.json is written on start and checked on resume — blocks silent config mixing.
  4. meta_<phase>.jsonl sidecars record token counts, truncation, and latency per generation.
  5. --out is refused outside /mnt/hdd (root fs on this box sits at 93%).

Usage:
  python run_labels.py --data /mnt/hdd/jonathan/data/tts_test.json \
                       --out /mnt/hdd/jonathan/runs/full_run
  python run_labels.py ... --end 20                 # smoke slice first (recommended)
  python run_labels.py ... --dtype fp16 --end 100   # FP16 comparison run (fresh --out!)
  python run_labels.py ... --phase retr_weak        # run a single phase only
Then:  python mint_labels.py --out <same --out> --data ...

CHANGELOG
  2026-07-28  smoke slice passed on the lab GPU box (4-bit, n=20, 8 solved)
  2026-07-29  determinism + provenance patch (items 1-5 above); pipeline semantics unchanged
"""
import os, sys, argparse, json, time, gc, random, hashlib, subprocess, datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

POOL_K = int(os.environ.get("SLR_POOL_K", 30))
MAXLEN = int(os.environ.get("SLR_MAXLEN", 4096))
WEAK_MODEL = os.environ.get("SLR_WEAK_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
RETR_OUT = int(os.environ.get("SLR_RETR_OUT", 768))
QA_OUT   = int(os.environ.get("SLR_QA_OUT", 128))
STRONG_MODEL = os.environ.get("SLR_STRONG_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SHUFFLE_SCHEME = "blake2b-8/2026-07-29"   # bump this string if the shuffle ever changes again
WRITE_ROOT = "/mnt/hdd/"                  # root fs is at 93% — never write outside this

# ---------------- data + pools (identical to pilot, except the seed) ----------------
def title_of(c): return c.split(':', 1)[0]

def stable_seed(s):
    """Deterministic across processes and machines.

    The previous version used Python's built-in hash(), which is salted with a random
    per-process value (PYTHONHASHSEED). That made the 'seeded' shuffle reshuffle the pool
    differently on every run, so pool ordering — and therefore which passages fell off the
    end at MAXLEN — was never reproducible. blake2b has no such salt.
    """
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)

def trim_pool(inst, k=POOL_K):
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
                        pool[j] = g; break
                have.add(t)
    rnd = random.Random(42 + stable_seed(inst['_id']) % 100000)
    rnd.shuffle(pool)
    return pool

# ---------------- prompts (AgentTTS recipe, identical to pilot) ----------------
SYS_RETR = "You are a helpful assistant that return the beneficial references to the query."

def retr_prompt(inst):
    refs = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(inst["ctxs_candidate"]))
    qr = f"Query: {inst['question']}\nReferences:\n{refs}"
    return (f"You have been provided with a query and a set of references. "
            f"Your task is to select references that comprehensively cover the information needed to answer the query. "
            f"Your response should include the original query and a list of selected references in the following format: "
            f"'Query: [query]\nSelected references: [reference number] [reference details]'."
            f"Here is the query and references: \n {qr}.")

def qa_prompt(inst, retrieved):
    return (f"Your task is to answer the query using the provided references. "
            f"If the references are insufficient, supplement your response with your own knowledge. "
            f"Answer with ONLY the exact answer phrase - no explanation, no full sentence.\n"
            f"Query: {inst['question']}\n{retrieved}")

import re
def parse_selection(txt, n_refs):
    part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
    nums = [int(x) for x in re.findall(r'\[(\d+)\]', part)]
    return sorted(set(n for n in nums if 1 <= n <= n_refs))

def build_block(inst, sel):
    if not sel: return f"Query: {inst['question']}\nSelected references: (none)"
    parts = [f"[{n}] {inst['ctxs_candidate'][n-1]}" for n in sel]
    return f"Query: {inst['question']}\nSelected references: " + " ".join(parts)

# ---------------- models ----------------
def load_model(which, dtype="4bit"):
    """weak is always fp16 (1.5B, no reason to quantize). dtype controls the STRONG model only."""
    if which == "weak":
        tok = AutoTokenizer.from_pretrained(WEAK_MODEL)
        m = AutoModelForCausalLM.from_pretrained(WEAK_MODEL,
                torch_dtype=torch.float16, device_map="cuda",
                attn_implementation="sdpa")
    else:
        tok = AutoTokenizer.from_pretrained(STRONG_MODEL)
        kw = dict(device_map="cuda", attn_implementation="sdpa")
        if dtype == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16
        m = AutoModelForCausalLM.from_pretrained(STRONG_MODEL, **kw)
    m.eval()
    return m, tok

def free(): gc.collect(); torch.cuda.empty_cache()

def generate(model, tok, system, user, max_new):
    """Returns (text, meta). meta is written to the sidecar log, never to the answer store."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXLEN).to(model.device)
    n_in = int(ids.input_ids.shape[1])
    meta = {"n_in": n_in, "truncated": bool(n_in >= MAXLEN)}
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        meta["n_out"] = int(out.shape[1]) - n_in
        meta["secs"] = round(time.time() - t0, 3)
        return tok.decode(out[0][n_in:], skip_special_tokens=True), meta
    except torch.cuda.OutOfMemoryError:
        free()
        meta.update({"n_out": 0, "secs": round(time.time() - t0, 3), "oom": True})
        return "[OOM_SKIPPED]", meta

# ---------------- append-only JSONL store (resume-safe) ----------------
class Store:
    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line); self.d[r['k']] = r['v']
                except Exception:
                    pass  # tolerate a torn final line from a crash
            print(f"  resumed {len(self.d)} entries from {os.path.basename(path)}")
        self.f = open(path, 'a')
    def __contains__(self, k): return k in self.d
    def __getitem__(self, k): return self.d[k]
    def put(self, k, v):
        self.d[k] = v
        self.f.write(json.dumps({'k': k, 'v': v}) + '\n')
    def flush(self): self.f.flush(); os.fsync(self.f.fileno())

class MetaLog:
    """Sidecar for per-generation diagnostics. Deliberately separate from Store so that
    mint_labels.py and every existing checkpoint file keep working unchanged."""
    def __init__(self, path):
        self.f = open(path, 'a')
    def put(self, k, meta):
        rec = {"k": k}; rec.update(meta)
        self.f.write(json.dumps(rec) + '\n')
    def flush(self): self.f.flush()

# ---------------- provenance ----------------
def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    def run(cmd):
        return subprocess.check_output(cmd, cwd=here, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        sha = run(["git", "rev-parse", "--short", "HEAD"])
        dirty = bool(run(["git", "status", "--porcelain"]))
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": None}

# these keys must match for a resume to be safe; anything else is free to differ
LOCKED = ["dtype", "pool_k", "maxlen", "weak_model", "strong_model", "shuffle_scheme", "data"]

def check_manifest(out_dir, args, force=False):
    path = os.path.join(out_dir, "run_manifest.json")
    current = {
        "dtype": args.dtype,
        "pool_k": POOL_K,
        "maxlen": MAXLEN,
        "weak_model": WEAK_MODEL,
        "strong_model": STRONG_MODEL,
        "shuffle_scheme": SHUFFLE_SCHEME,
        "data": os.path.abspath(args.data),
    }
    if os.path.exists(path):
        prev = json.load(open(path))
        bad = [k for k in LOCKED if prev.get(k) != current[k]]
        if bad and not force:
            lines = "\n".join(f"    {k}: existing={prev.get(k)!r}  requested={current[k]!r}"
                              for k in bad)
            raise SystemExit(
                f"\nCONFIG MISMATCH — refusing to resume into {out_dir}\n{lines}\n\n"
                f"  Results in this directory were produced with a different configuration.\n"
                f"  Resuming would silently mix them. Use a FRESH --out directory.\n"
                f"  (Override with --force only if you are certain they are comparable.)\n")
        print(f"manifest OK — resuming a {prev.get('dtype')} run started {prev.get('created')}")
        return
    stale = [f for f in os.listdir(out_dir) if f.endswith(".jsonl")]
    if stale and not force:
        raise SystemExit(
            f"\n{out_dir} holds results from before manifests existed:\n"
            f"    {', '.join(sorted(stale))}\n\n"
            f"  Those were generated with the old (non-deterministic) pool shuffle and cannot\n"
            f"  be pooled with new output. Use a FRESH --out directory. Override with --force.\n")
    current.update(git_info())
    current.update({
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "argv": sys.argv[1:],
        "forced": bool(force),
    })
    json.dump(current, open(path, "w"), indent=2)
    print(f"wrote manifest -> {path}  (git {current['git_sha']}"
          f"{', DIRTY' if current['git_dirty'] else ''})")

# ---------------- phases ----------------
def run_retrieval(which, data, out_dir, dtype):
    store = Store(os.path.join(out_dir, f"retr_{which}.jsonl"))
    meta = MetaLog(os.path.join(out_dir, f"meta_retr_{which}.jsonl"))
    todo = [(i, inst) for i, inst in data if str(i) not in store]
    if not todo:
        print(f"[retr_{which}] fully cached ({len(data)})"); return store
    print(f"[retr_{which}] {len(data)-len(todo)} cached, {len(todo)} to run")
    m, tok = load_model(which, dtype)
    print(f"[retr_{which}] model loaded ({dtype if which=='strong' else 'fp16'}) | "
          f"GPU free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    t0 = time.time(); done = 0; n_trunc = 0
    for i, inst in todo:
        txt, mt = generate(m, tok, SYS_RETR, retr_prompt(inst), RETR_OUT)
        store.put(str(i), txt); meta.put(str(i), mt)
        n_trunc += mt["truncated"]
        if txt.startswith('[OOM'):
            store.flush(); meta.flush()
            raise RuntimeError(f"OOM at retr_{which} idx {i} — stopping (fail-fast). "
                               f"GPU free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
        done += 1
        if done % 25 == 0:
            store.flush(); meta.flush()
            rate = (time.time()-t0)/done
            eta_h = rate*(len(todo)-done)/3600
            print(f"  retr_{which} {done}/{len(todo)}  {rate:.1f}s/inst  ETA {eta_h:.1f}h  "
                  f"truncated {n_trunc}/{done}", flush=True)
    store.flush(); meta.flush(); del m, tok; free()
    print(f"[retr_{which}] DONE ({(time.time()-t0)/3600:.2f}h) | "
          f"prompts truncated at {MAXLEN} tokens: {n_trunc}/{done}")
    return store

def run_qa(which, data, retr, out_dir, dtype):
    store = Store(os.path.join(out_dir, f"qa_{which}.jsonl"))
    meta = MetaLog(os.path.join(out_dir, f"meta_qa_{which}.jsonl"))
    blocks = {}
    for i, inst in data:
        for rsrc in ("weak", "strong"):
            sel = parse_selection(retr[rsrc][str(i)], len(inst['ctxs_candidate']))
            blocks[(rsrc, i)] = build_block(inst, sel)
    todo = [(i, inst, rsrc) for i, inst in data for rsrc in ("weak", "strong")
            if f"{i}|{rsrc}" not in store]
    if not todo:
        print(f"[qa_{which}] fully cached"); return store
    print(f"[qa_{which}] {len(todo)} generations to run")
    m, tok = load_model(which, dtype)
    print(f"[qa_{which}] model loaded ({dtype if which=='strong' else 'fp16'}) | "
          f"GPU free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    t0 = time.time(); done = 0; n_trunc = 0
    for i, inst, rsrc in todo:
        txt, mt = generate(m, tok, "You are a helpful assistant.",
                           qa_prompt(inst, blocks[(rsrc, i)]), QA_OUT)
        store.put(f"{i}|{rsrc}", txt); meta.put(f"{i}|{rsrc}", mt)
        n_trunc += mt["truncated"]
        if txt.startswith('[OOM'):
            store.flush(); meta.flush()
            raise RuntimeError(f"OOM at qa_{which} {i}|{rsrc} — stopping (fail-fast).")
        done += 1
        if done % 50 == 0:
            store.flush(); meta.flush()
            rate = (time.time()-t0)/done
            print(f"  qa_{which} {done}/{len(todo)}  {rate:.1f}s/gen  "
                  f"ETA {rate*(len(todo)-done)/3600:.1f}h  truncated {n_trunc}/{done}", flush=True)
    store.flush(); meta.flush(); del m, tok; free()
    print(f"[qa_{which}] DONE ({(time.time()-t0)/3600:.2f}h) | "
          f"prompts truncated at {MAXLEN} tokens: {n_trunc}/{done}")
    return store

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='path to tts_test.json (7,405-instance pool file)')
    ap.add_argument('--out', required=True, help='output/checkpoint directory (must be under /mnt/hdd)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None, help='exclusive; use --end 20 for a smoke slice')
    ap.add_argument('--dtype', default='4bit', choices=['4bit', 'fp16'],
                    help='precision of the STRONG (7B) model; weak is always fp16')
    ap.add_argument('--phase', default='all',
                    choices=['all', 'retr_weak', 'retr_strong', 'qa_weak', 'qa_strong'])
    ap.add_argument('--force', action='store_true',
                    help='override the manifest guard (you should almost never need this)')
    args = ap.parse_args()

    out_abs = os.path.abspath(args.out)
    if not out_abs.startswith(WRITE_ROOT):
        raise SystemExit(f"--out must live under {WRITE_ROOT} — the root filesystem on this "
                         f"machine is at 93% and is shared. Got: {out_abs}")
    os.makedirs(args.out, exist_ok=True)

    print("PID:", os.getpid())
    assert torch.cuda.is_available(), "no CUDA device visible"
    free_gb = torch.cuda.mem_get_info()[0]/1e9
    need_gb = 20 if args.dtype == 'fp16' else 13
    print(f"GPU: {torch.cuda.get_device_name(0)} | free {free_gb:.1f} GB | "
          f"strong model dtype: {args.dtype} (needs ~{need_gb} GB)")
    assert free_gb > need_gb, (f"GPU not clean — {free_gb:.1f} GB free, need >{need_gb} for "
                               f"--dtype {args.dtype}. Is someone else's job running? Check nvidia-smi.")

    check_manifest(args.out, args, force=args.force)

    print(f"loading {args.data} ...")
    hotpot = json.load(open(args.data))
    end = args.end if args.end is not None else len(hotpot)
    raw = hotpot[args.start:end]
    print(f"{len(raw)} instances selected [{args.start}:{end}] of {len(hotpot)}")

    data = []
    covered = 0
    for off, inst in enumerate(raw):
        inst = dict(inst)
        inst['ctxs_candidate'] = trim_pool(inst)
        golds = set(inst.get('supporting_titles', []))
        if golds <= set(title_of(c) for c in inst['ctxs_candidate']):
            covered += 1
        data.append((args.start + off, inst))
    print(f"pools trimmed to {POOL_K}; gold fully in pool: {covered}/{len(data)}")

    retr = {}
    if args.phase in ('all', 'retr_weak'):
        retr['weak'] = run_retrieval('weak', data, args.out, args.dtype)
    if args.phase in ('all', 'retr_strong'):
        retr['strong'] = run_retrieval('strong', data, args.out, args.dtype)
    if args.phase in ('all', 'qa_weak', 'qa_strong'):
        if 'weak' not in retr:  retr['weak']  = Store(os.path.join(args.out, "retr_weak.jsonl"))
        if 'strong' not in retr: retr['strong'] = Store(os.path.join(args.out, "retr_strong.jsonl"))
        missing = [i for i, _ in data if str(i) not in retr['weak'] or str(i) not in retr['strong']]
        assert not missing, f"retrieval incomplete for {len(missing)} instances (e.g. idx {missing[:5]}) — run retrieval phases first"
    if args.phase in ('all', 'qa_weak'):
        run_qa('weak', data, retr, args.out, args.dtype)
    if args.phase in ('all', 'qa_strong'):
        run_qa('strong', data, retr, args.out, args.dtype)
    print("ALL REQUESTED PHASES COMPLETE. Next: python mint_labels.py --out", args.out, "--data", args.data)

if __name__ == '__main__':
    main()
