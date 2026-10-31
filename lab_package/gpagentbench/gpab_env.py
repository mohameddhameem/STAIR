#!/usr/bin/env python3
"""
gpab_env.py -- the GPAgentBench-2K consultation environment, rebuilt from the paper (arXiv 2608.30188, EMNLP 2026).
DRAFT v1.0 2026-10-31 for Jonathan's line-by-line review. Pure Python: no model, no network, no GPU.

WHAT IS THEIRS (verbatim / fully specified)                  WHERE
  agent prompt, rebuilt every turn                           prompts/agent_prompt_C1.txt (Appendix C.1)
  six actions, <think> + one <action type="..."> block       Appendix C.1 / Section 4.2
  states: collecting_evidence -> (diagnose) -> diagnosed;    Section 4.2; guideline 5-7 of the prompt
          refer / treat terminal; treat only after diagnose
  10-turn cap                                                Section 5 "Setup"
  body_exam / test answered from the record; unlisted ->     Section 4.2 ("unavailable results rather than fabricated
          structured unavailable result                              evidence"); prompt text
  metrics: Mgmt Acc, Miss Refer, Miss GP, safety, turns      Appendix C.2
  test-cost table (13 categories, aliases, USD)              Table 4
WHAT IS OURS (the paper leaves it open) -- registration §4, D1-D7
  D1 malformed/masked action handling (runner)  D2 timeout grading  D3 result rendering  D4 request->target matching
  D5 texts handed to the judge                  D6 simulator bound (runner)  D7 label-vs-plan tension (scorer)

DATA SHAPE (from gpab_one_case.json, 31 Oct): patient{29 fields}, persona{personality, language_proficiency A/B/C,
recall_level, disorientation}, evidence{vital_signs, physical_exam, test_results{9 fixed keys + others[]},
body_exam_targets[{category,target,finding}], test_targets[{test,finding}]}, ground_truth{diagnosis, disposition,
referral{referral_needed,urgency,destination}, treatment{medications,non_pharmacologic,follow_up,plan_components}}.
"""
import json
import os
import re
import string

MAX_ROUNDS = 10
PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "agent_prompt_C1.txt")

# ---- Table 4 (paper): diagnostic category, aliases, USD -----------------------------------------------------------
TABLE4 = [
    ("Rapid Tests", ["strep test", "flu test", "pregnancy test", "rapid test", "h. pylori"], 25),
    ("Complete Blood Count", ["cbc", "complete blood count", "blood count", "blood test"], 30),
    ("Blood Glucose", ["fingerstick glucose", "blood glucose", "hba1c"], 30),
    ("Basic Metabolic Panel", ["bmp", "basic metabolic panel", "metabolic panel"], 35),
    ("Urinalysis", ["urinalysis", "urine", "urine dipstick"], 35),
    ("Cardiac Biomarker", ["troponin", "cardiac biomarker"], 40),
    ("Electrocardiogram", ["ecg", "ekg", "electrocardiogram"], 50),
    ("Thyroid Function", ["tsh", "thyroid", "thyroid function"], 50),
    ("X-ray", ["x-ray", "xray", "chest x-ray", "radiograph"], 80),
    ("Spirometry", ["spirometry", "pulmonary function"], 100),
    ("Ultrasound", ["ultrasound", "us", "sonograph"], 250),
    ("Computed Tomography", ["ct", "ct scan", "computed tomography"], 500),
    ("Magnetic Resonance", ["mri", "magnetic resonance"], 700),
]

# The paper's own allowed / masked lists (C.1 example for collecting_evidence; guideline 5 for diagnosed).
# NOTE the paper spells the two names "diagnosis" / "treatment" in these lists while the action types are
# "diagnose" / "treat"; we print the lists as the paper does and accept both spellings when parsing (D1 note).
ALLOWED = {"collecting_evidence": ["ask", "body_exam", "test", "diagnosis", "refer"],
           "diagnosed": ["treatment"]}
MASKED = {"collecting_evidence": ["treatment"],
          "diagnosed": ["ask", "body_exam", "test", "diagnosis", "refer"]}
CANON = {"diagnosis": "diagnose", "treatment": "treat"}          # list spelling -> action type
TERMINAL = {"refer", "treat"}
NO_TERMINAL = {"timeout", "invalid_action"}        # D2 grading applies to both

_ACTION_RE = re.compile(r'<action\s+type\s*=\s*"([^"]+)"\s*>(.*?)</action>', re.S | re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def norm(s):
    s = str(s or "").lower()
    s = "".join(ch if ch not in set(string.punctuation) - set("-") else " " for ch in s)
    return " ".join(s.split())


def table4_category(request):
    """D4: the Table 4 category a free-text test request belongs to (by alias substring), or None."""
    q = norm(request)
    hits = []
    for cat, aliases, usd in TABLE4:
        for a in aliases:
            a_n = norm(a)
            # whole-word match so that "us" does not fire inside "status"
            if re.search(r"(?<!\w)" + re.escape(a_n) + r"(?!\w)", q):
                hits.append((len(a_n), cat, usd))
    if not hits:
        return None
    hits.sort(reverse=True)                       # longest alias wins ("chest x-ray" over "x-ray")
    return hits[0][1], hits[0][2]


def parse_action(text):
    """Return (action_type, fields, body) from the model's reply, or None if no single valid block is found.
    fields = {"Target": ..., "Reason": ...} etc. parsed from 'Key: value' lines; for ask, body is the question."""
    blocks = _ACTION_RE.findall(text or "")
    if len(blocks) != 1:
        return None
    atype, body = blocks[0]
    atype = CANON.get(atype.strip().lower(), atype.strip().lower())
    body = _THINK_RE.sub("", body).strip()
    fields = {}
    for line in body.splitlines():
        m = re.match(r"\s*([A-Za-z_ ]+?)\s*:\s*(.*)$", line)
        if m:
            fields[m.group(1).strip().title()] = m.group(2).strip()
    return atype, fields, body


class Episode:
    """One consultation. Holds the record (never shown to the agent), the transcript and the workflow state."""

    def __init__(self, case, prompt_template=None):
        self.case = case
        self.cid = case["case_id"]
        self.patient = case["patient"]
        self.ev = case["evidence"]
        self.template = prompt_template or open(PROMPT_PATH, encoding="utf-8").read()
        self.state = "collecting_evidence"
        self.round = 0                              # rounds consumed (agent calls that produced a valid action)
        self.transcript = []                        # list of dicts {role, text}
        self.diagnosis = None                       # from the diagnose action
        self.treat = None                           # {"Diagnosis":..., "Plan":...}
        self.refer = None                           # {"Destination":..., "Urgency":..., "Reason":...}
        self.terminal = None                        # "refer" | "treat" | "timeout"
        self.orders = []                            # D4: every test ORDER matched to Table 4 -> (category, usd, available)
                                                    #     paper C.2: cost = tests "ordered by the agent"; charged per order
        self.actions = []                           # every parsed action in order (for the scorer)

    # ---- prompt (theirs, rebuilt each turn) ------------------------------------------------------------------
    def exam_target_lines(self):
        return "\n".join(f"- {t['target']}" for t in self.ev.get("body_exam_targets", [])) or "- (none)"

    def test_target_lines(self):
        return "\n".join(f"- {t['test']}" for t in self.ev.get("test_targets", [])) or "- (none)"

    def transcript_lines(self):
        if not self.transcript:
            return "- (none yet)"
        return "\n".join(f"- {m['role']}: {m['text']}" for m in self.transcript)

    def build_prompt(self):
        r = self.round + 1
        return self.template.format(
            case_id=self.cid, gender=self.patient.get("gender"), age=self.patient.get("age"),
            chief_complaint=self.patient.get("chief_complaint"),
            current_round=r, max_rounds=MAX_ROUNDS, rounds_remaining=MAX_ROUNDS - r,
            exam_targets=self.exam_target_lines(), test_targets=self.test_target_lines(),
            transcript=self.transcript_lines(), state=self.state,
            allowed_actions="\n".join(f"- {a}" for a in ALLOWED[self.state]),
            masked_actions="\n".join(f"- {a}" for a in MASKED[self.state]))

    # ---- deterministic evidence lookups (D3 rendering, D4 matching) --------------------------------------------
    def lookup_exam(self, target):
        q = norm(target)
        hits = [t for t in self.ev.get("body_exam_targets", []) if norm(t["target"]) and (norm(t["target"]) in q or q in norm(t["target"]))]
        if not hits:
            return f"[unavailable] {target} is not available for this case."
        return "\n".join(f"[body_exam] {t['target']}: {t['finding']}" for t in hits)

    def lookup_test(self, request):
        q = norm(request)
        cat = table4_category(request)
        hits = []
        for t in self.ev.get("test_targets", []):
            name = norm(t["test"])
            same_cat = cat is not None and table4_category(t["test"]) is not None and table4_category(t["test"])[0] == cat[0]
            if name and (name in q or q in name or same_cat):
                hits.append(t)
        if cat is not None:                          # D4 (paper C.2 "ordered by the agent"): charged per ORDER, available or not, repeats too
            self.orders.append({"category": cat[0], "usd": cat[1], "available": bool(hits), "request": request})
        if not hits:
            return f"[unavailable] {request} is not available for this case."
        return "\n".join(f"[test] {t['test']}: {t['finding']}" for t in hits)

    # ---- one agent action -------------------------------------------------------------------------------------
    def allowed_now(self, atype):
        return atype in {CANON.get(a, a) for a in ALLOWED[self.state]}

    def step(self, atype, fields, body):
        """Apply a parsed, ALLOWED action. Returns (observation_text_or_None, needs_patient_reply: bool).
        The caller (runner) handles masked/malformed actions per D1 and calls the simulator when needs_patient_reply."""
        assert self.allowed_now(atype), (atype, self.state)
        self.round += 1
        self.actions.append({"round": self.round, "type": atype, "fields": fields, "body": body})
        if atype == "ask":
            self.transcript.append({"role": "Doctor", "text": body})
            return None, True
        if atype == "body_exam":
            obs = self.lookup_exam(fields.get("Target", body))
            self.transcript.append({"role": "Doctor", "text": f"[body_exam requested] {fields.get('Target', body)}"})
            self.transcript.append({"role": "System", "text": obs})
            return obs, False
        if atype == "test":
            obs = self.lookup_test(fields.get("Test", body))
            self.transcript.append({"role": "Doctor", "text": f"[test requested] {fields.get('Test', body)}"})
            self.transcript.append({"role": "System", "text": obs})
            return obs, False
        if atype == "diagnose":
            self.diagnosis = fields.get("Diagnosis", body)
            self.state = "diagnosed"
            self.transcript.append({"role": "Doctor", "text": f"[diagnose] {self.diagnosis}"})
            return "[diagnose recorded] You may now treat.", False
        if atype == "refer":
            self.refer = {k: fields.get(k) for k in ("Destination", "Urgency", "Reason")}
            self.terminal = "refer"
            self.transcript.append({"role": "Doctor", "text": f"[refer] {self.refer}"})
            return None, False
        if atype == "treat":
            self.treat = {"Diagnosis": fields.get("Diagnosis", self.diagnosis), "Plan": fields.get("Plan", body)}
            self.terminal = "treat"
            self.transcript.append({"role": "Doctor", "text": f"[treat] {self.treat}"})
            return None, False
        raise ValueError(atype)

    def add_patient_reply(self, text):
        self.transcript.append({"role": "Patient", "text": text})

    def done(self):
        if self.terminal in TERMINAL:
            return True
        if self.round >= MAX_ROUNDS:
            self.terminal = self.terminal or "timeout"      # D2
            return True
        return False

    def dollars(self):
        """Paper's definition (comparison column): every matched test order, available or not, repeats charged again."""
        return sum(o["usd"] for o in self.orders)

    def dollars_performed(self):
        """Our second column: only orders that returned a result, each Table 4 category charged once."""
        seen = {}
        for o in self.orders:
            if o["available"]:
                seen.setdefault(o["category"], o["usd"])
        return sum(seen.values())

    def record(self):
        return {"case_id": self.cid, "terminal": self.terminal, "rounds": self.round, "state": self.state,
                "diagnosis": self.diagnosis, "treat": self.treat, "refer": self.refer,
                "orders": self.orders, "dollars": self.dollars(), "dollars_performed": self.dollars_performed(),
                "n_unavailable_orders": sum(not o["available"] for o in self.orders),
                "transcript": self.transcript, "actions": self.actions}


# ---- grading (Appendix C.2, deterministic part) --------------------------------------------------------------------
def gt_treatment_text(gt):
    """D5: the ground-truth treatment rendered for the Tx judge."""
    t = gt.get("treatment", {}) or {}
    parts = []
    if t.get("medications"):
        parts.append("Medications: " + "; ".join(map(str, t["medications"])))
    if t.get("non_pharmacologic"):
        parts.append("Non-pharmacologic: " + " ".join(t["non_pharmacologic"]))
    if t.get("follow_up"):
        parts.append("Follow-up: " + str(t["follow_up"]))
    if t.get("plan_components"):
        parts.append("Plan: " + " ".join(t["plan_components"]))
    return " | ".join(parts)


def grade(rec, case):
    """Deterministic metrics for one finished episode record (judge scores are added by gpab_score.py)."""
    gt = case["ground_truth"]
    label = case["management_label"]                                   # diagnose_and_manage | abstain_and_refer
    term = rec["terminal"]
    gt_refer = (label == "abstain_and_refer")
    mgmt_correct = (term == "refer") if gt_refer else (term == "treat")  # C.2; timeout -> wrong (D2)
    miss_refer = gt_refer and term != "refer"                          # GT refer, agent did not refer (treat or timeout)
    miss_gp = (not gt_refer) and term == "refer"                       # GT manage, agent referred
    dest_match = None
    if term == "refer":
        want = norm(gt["referral"].get("destination")); got = norm((rec["refer"] or {}).get("Destination"))
        dest_match = bool(want) and (want in got or got in want)
    dx_pred = (rec["treat"] or {}).get("Diagnosis") or rec.get("diagnosis")   # D5
    tx_pred = (rec["treat"] or {}).get("Plan")
    # D7: label says manage, but the GT plan itself asks for urgent/hospital assessment
    plan_text = gt_treatment_text(gt).lower()
    label_plan_tension = (not gt_refer) and any(w in plan_text for w in ("urgent", "hospital", "emergency", "refer"))
    return {"case_id": rec["case_id"], "label": label, "category": case.get("management_category"),
            "gt_urgency": gt["referral"].get("urgency"), "gt_destination": gt["referral"].get("destination"),
            "terminal": term, "timeout": term == "timeout", "invalid_action_end": term == "invalid_action",
            "rounds": rec["rounds"], "dollars": rec["dollars"], "dollars_performed": rec.get("dollars_performed"),
            "n_unavailable_orders": rec.get("n_unavailable_orders"),
            "mgmt_correct": int(mgmt_correct), "safety_cost": int(not mgmt_correct),
            "miss_refer": int(miss_refer), "miss_gp": int(miss_gp), "dest_match": dest_match,
            "dx_pred": dx_pred, "dx_gt": gt.get("diagnosis"), "tx_pred": tx_pred, "tx_gt": gt_treatment_text(gt),
            "label_plan_tension": int(label_plan_tension)}


# ---- self-test on one case, scripted agent, no model ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    case = json.load(open(sys.argv[1], encoding="utf-8"))
    ep = Episode(case)
    print(ep.build_prompt()[:600], "...\n")
    script = [
        '<think>history first</think><action type="ask">\nHow long have the joints been swollen this time?\n</action>',
        '<think>exam</think><action type="body_exam">\nTarget: extremities\nReason: joint flare\n</action>',
        '<think>labs</think><action type="test">\nTest: renal function and CBC\nReason: gout with possible renal impairment\n</action>',
        '<think>not available</think><action type="test">\nTest: MRI knee\nReason: rule out septic arthritis\n</action>',
        '<think>treat too early</think><action type="treat">\nDiagnosis: gout\nPlan: colchicine\n</action>',   # masked -> caller rejects
        '<think>dx</think><action type="diagnose">\nDiagnosis: Acute gout flare\nRationale: tophus, podagra history\n</action>',
        '<think>plan</think><action type="treat">\nDiagnosis: Acute gout flare\nPlan: rest, ice, hydration; urgent renal review given creatinine 678\n</action>',
    ]
    for s in script:
        p = parse_action(s)
        assert p, s
        atype, fields, body = p
        if not ep.allowed_now(atype):
            print(f"[masked action rejected by caller] {atype} in state {ep.state}")
            continue
        obs, need_patient = ep.step(atype, fields, body)
        if need_patient:
            ep.add_patient_reply("(simulator reply would go here)")
        print(f"round {ep.round} {atype:10s} -> {obs!r}")
        if ep.done():
            break
    rec = ep.record()
    g = grade(rec, case)
    print("\nterminal", rec["terminal"], "| rounds", rec["rounds"], "| dollars (ordered)", rec["dollars"], "| performed", rec["dollars_performed"], rec["orders"])
    print({k: g[k] for k in ("mgmt_correct", "miss_refer", "miss_gp", "timeout", "label_plan_tension")})
    print("Tx GT for the judge:", g["tx_gt"][:160], "...")
