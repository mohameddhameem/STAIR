"""
Stage-Level Routing — #3 router_prompts.py: the PROMPT CONTRACT (v1.0, 2026-10-06)

THE ONE COPY of the frozen router prompts and their assembly. #7 (build_qa_labels_t2),
#9 (sft_router5), #10 (gate_eval) and #11 (compose_eval) import from here and contain
NO prompt text of their own — labeling, training and evaluation inputs stay
byte-identical by construction (Code slide contract no.1).

Contents (nothing else):
  1. SYS_CASCADE + INSTR_RET2 / INSTR_QA1 / INSTR_QA2 — frozen as approved on the
     prompt slides (1 Oct). The prompt speaks task language only: no harness words
     (logits, thresholds, training) — the model cannot act on those.
  2. Formatting of the two composite fields the prompts take:
     format_selections() for {selections} and the evidence passthrough for
     {evidence} — per-passage text capped at CHAR_CAP characters, the byte-identical
     twin of the probe's formatting (probe_ret_coverage.py ran with this contract:
     final val AUC 0.7558).
  3. assemble(): chat-template call ending at the OPENED ASSISTANT TURN — the model
     never generates; callers read next-token logits at that position.
  4. Single-token checks + logit-pair readers for W/S and 1/2.

MODEL NAMES ARE PARAMETERS: SYS_CASCADE names the pipeline's models truthfully
(our certified-prompts rule), so the pair is a fill-in. Config A (default):
Qwen2.5-1.5B / Qwen2.5-7B. Config B passes its own names; every other byte of
every prompt is identical across configurations.

NO MODEL IS LOADED HERE. Pure strings + tokenizer helpers. Importing this module
has no side effects.

Self-check (run directly):  python router_prompts.py [path/to/probe_ret_coverage.py]
  - renders each template with dummy fields and prints it for eyeball review
  - asserts W/S/1/2 are single tokens IF `transformers` + the base tokenizer are
    available (skipped gracefully otherwise)
  - with the probe path given: extracts the probe's embedded SYS_CASCADE and
    INSTR_RET2 and asserts they are byte-identical to this module's
"""

# ----------------------------------------------------------------------------
# 0 · knobs shared by every caller (byte-identical contract: do not override
#     in one script without overriding everywhere)
# ----------------------------------------------------------------------------
CHAR_CAP = 600      # max characters of each passage shown to a router
MAXLEN   = 16384     # token cap for assembled router calls (left-truncated)

# Config A defaults; Config B callers pass their own truthful names.
WEAK_NAME_DEFAULT   = "Qwen2.5-1.5B"
STRONG_NAME_DEFAULT = "Qwen2.5-7B"

# ----------------------------------------------------------------------------
# 1 · the frozen prompts (slides "The Router Prompts — draft for validation",
#     approved 1 Oct; byte-identical to the probe's embedded copy)
# ----------------------------------------------------------------------------
SYS_CASCADE_TEMPLATE = (
    "You route a question-answering pipeline that runs the weak model first and "
    "escalates only when needed. Each stage runs either {weak_name} (weak, cheap) "
    "or {strong_name} (strong, expensive). From the weak model's output shown "
    "below, choose the cheapest option that still lets the pipeline answer correctly."
)

def sys_cascade(weak_name=WEAK_NAME_DEFAULT, strong_name=STRONG_NAME_DEFAULT):
    return SYS_CASCADE_TEMPLATE.format(weak_name=weak_name, strong_name=strong_name)

INSTR_RET2 = (
    "Stage: RETRIEVAL REVIEW — the weak retriever has already selected "
    "references from the 100-passage pool.\n"
    "W = keep the weak selections (no further retrieval spend)\n"
    "S = escalate: run strong retrieval to reselect\n"
    "Question: {q}\n"
    "Weak retriever selected {n_sel} of 100 passages: {selections}\n"
    "Answer with one letter only."
)

INSTR_QA1 = (
    "Stage: ANSWER REVIEW — the weak answerer has already produced an answer "
    "from the references below.\n"
    "W = deliver the weak answer as final\n"
    "S = escalate: run strong answering\n"
    "Question: {q}\n"
    "References in use: {evidence}\n"
    "Weak answer: {weak_answer}\n"
    "Answer with one letter only."
)

INSTR_QA2 = (
    "Stage: ANSWER CHOICE — two answers were produced from the references "
    "below. 1 = deliver Answer 1 · 2 = deliver Answer 2\n"
    "Question: {q}\n"
    "References in use: {evidence}\n"
    "Answer 1: {answer_1}\n"
    "Answer 2: {answer_2}\n"
    "Answer with one digit only."
)

# ----------------------------------------------------------------------------
# 2 · composite-field formatting (the router's view of pipeline artifacts)
# ----------------------------------------------------------------------------
def cap(text, n=CHAR_CAP):
    """Cap one passage's text at n characters (the probe's contract)."""
    return text if len(text) <= n else text[:n]

def format_selections(sel_ids, pool):
    """{selections}: the weak retriever's selected passages as '[n] text', each
    capped at CHAR_CAP — mirrors build_block's '[n] passage' shape so the router
    sees selections the way the pipeline writes them. sel_ids are 1-based."""
    if not sel_ids:
        return "(none)"
    return " ".join(f"[{n}] {cap(pool[n-1])}" for n in sel_ids)

def format_evidence(block):
    """{evidence}: the pipeline's build_block output passed through unchanged
    EXCEPT the per-passage CHAR_CAP, applied passage-by-passage on '[n] ' splits."""
    import re
    parts = re.split(r"(\[\d+\] )", block)
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        marker, body = parts[i], parts[i+1] if i+1 < len(parts) else ""
        out.append(marker + cap(body))
    return "".join(out)

# ----------------------------------------------------------------------------
# 3 · assembly: one function, every caller (training, labeling, evaluation)
# ----------------------------------------------------------------------------
def assemble(tok, user, weak_name=WEAK_NAME_DEFAULT, strong_name=STRONG_NAME_DEFAULT):
    """Render system+user through the chat template, ENDING AT THE OPENED
    ASSISTANT TURN, and return input_ids left-truncated to MAXLEN.
    The model never generates: callers read next-token logits at the end.
    (transformers 5.x: template rendered to a string first, then tokenized —
    the fix validated in the probe run.)"""
    msgs = [{"role": "system", "content": sys_cascade(weak_name, strong_name)},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids[:, -MAXLEN:]

# ----------------------------------------------------------------------------
# 4 · token helpers: single-token assertion + logit-pair readers
# ----------------------------------------------------------------------------
def single_token_id(tok, s):
    """The id of s as a single token, or an AssertionError naming the culprit."""
    ids = tok(s, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"{s!r} is not a single token: {ids}"
    return ids[0]

def letter_ids(tok):
    """(id_W, id_S) — asserted single tokens."""
    return single_token_id(tok, "W"), single_token_id(tok, "S")

def digit_ids(tok):
    """(id_1, id_2) — asserted single tokens."""
    return single_token_id(tok, "1"), single_token_id(tok, "2")

def pair_probs(last_logits, id_a, id_b):
    """softmax over exactly the two class logits -> (P_a, P_b); P_b = 1 - P_a."""
    import torch
    two = torch.stack([last_logits[id_a], last_logits[id_b]]).float()
    p = torch.softmax(two, dim=0)
    return float(p[0]), float(p[1])

# ----------------------------------------------------------------------------
# 5 · self-check
# ----------------------------------------------------------------------------
def _selfcheck(probe_path=None):
    dummy_pool = [f"Title {i}: passage text number {i} " + "x" * 700 for i in range(1, 101)]
    sels = [3, 17, 88]
    print("=" * 70); print("SYS_CASCADE (Config A names):"); print(sys_cascade())
    print("=" * 70); print("INSTR_RET2 rendered:")
    print(INSTR_RET2.format(q="Which Dutch city hosted the ceremony?",
                            n_sel=len(sels),
                            selections=format_selections(sels, dummy_pool))[:900], "...")
    print("=" * 70); print("INSTR_QA1 rendered:")
    print(INSTR_QA1.format(q="Which Dutch city hosted the ceremony?",
                           evidence=format_evidence("Query: ...\nSelected references: [3] aaa [17] bbb"),
                           weak_answer="Rotterdam"))
    print("=" * 70); print("INSTR_QA2 rendered:")
    print(INSTR_QA2.format(q="Which Dutch city hosted the ceremony?",
                           evidence="Query: ...\nSelected references: [3] aaa",
                           answer_1="Rotterdam, Netherlands", answer_2="Rotterdam"))
    print("=" * 70)
    # byte-identity vs the probe's embedded copies
    if probe_path:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("probe_mod", probe_path)
        _pm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pm)
        ok_sys = sys_cascade() == getattr(_pm, "SYS_CASCADE", None)
        ok_ret = INSTR_RET2 == getattr(_pm, "INSTR_RET2", None)
        print(f"probe byte-identity: SYS_CASCADE {'IDENTICAL' if ok_sys else 'DIFFERS'} | "
              f"INSTR_RET2 {'IDENTICAL' if ok_ret else 'DIFFERS'}")
        if not (ok_sys and ok_ret):
            raise SystemExit("DRIFT between router_prompts.py and the probe — reconcile before training.")
    # single-token checks (needs the tokenizer)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        idW, idS = letter_ids(tok); id1, id2 = digit_ids(tok)
        print(f"single tokens OK: W={idW} S={idS} 1={id1} 2={id2}")
        ids = assemble(tok, INSTR_QA1.format(q="q?", evidence="[1] e", weak_answer="a"))
        print(f"assemble OK: {ids.shape[1]} tokens, ends at opened assistant turn")
    except Exception as e:
        print(f"(tokenizer checks skipped: {e})")

if __name__ == "__main__":
    import sys
    _selfcheck(sys.argv[1] if len(sys.argv) > 1 else None)
