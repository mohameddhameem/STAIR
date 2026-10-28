# MedAgentBench Campaign — Pre-Registration (DRAFT for Jonathan's review, 22 Oct 2026)

Status: DRAFT. Items marked **[DECIDED, date]** were settled in discussion; items marked **[OPEN]**
require a decision before this document is frozen. Nothing launches until the document is frozen
and the code and prompts are validated line-by-line.

---

## 1. Objective and registered claims

Train the stage-level router with GRPO(Reward) on MedAgentBench — the first dataset with **no labels
and no cached grid** — and test two registered claims:

1. **Transfer:** the validated recipe (GRPO, A1 mean-baseline advantage) reaches a useful
   cost–accuracy operating point on a live-execution benchmark. Transfer baseline = the HotpotQA
   GRPO both-routers held-out results (76.23 @ 80.5% full test / 78.10 @ 79.0% n=105 / 75.80 @ 84.3%
   random-500) **[DECIDED, 21 Oct]**.
2. **Risk contribution:** risk-aware routing (the τ ≥ 0.8 hard gate) measurably changes where
   failures land, at a reported cost premium — the with/without comparison of §5.

## 2. Benchmark and metrics

- **Benchmark:** MedAgentBench (arXiv 2501.14654) — 300 tasks, 10 categories × 30, simulated FHIR
  server; 150 query (GET) tasks graded against curated references, 150 action (POST) tasks graded by
  rule-based payload checks; invalid action or > 8 interaction rounds = failure.
- **Accuracy metric: the benchmark's own success rate (SR), pass@1, binary** — no partial credit
  **[DECIDED, 22 Oct]**. Q in the reward = this binary SR.
- **Cost metric: ours, registered here** — the paper reports no cost metric, so the cost axis is our
  measured per-pass token costs with C_SS = the measured always-strong cost (budget ≡ C_SS, the
  21 Oct decision). **[OPEN: measure and freeze the per-model path costs during calibration, §7.]**
- **Supplementary metrics (declared):** (a) failure breakdown by risk class — three outcomes,
  now **defined mechanically by the released graders (refsol.py, read in full 22 Oct)**:
  **correct** = grader returns True · **incorrect but fixable** = grader False with no accepted
  POST in the episode history · **incorrect, not fixable** = grader False with an accepted POST
  (a wrong order actually submitted: wrong dose, wrong code, or a POST on a branch requiring none).
  No judgment calls — read straight off their artifacts. Note: the benchmark itself is risk-aligned —
  **every query task fails automatically if the agent makes ANY POST** (refsol check_has_post), so
  "wrong executed actions" extends a principle their graders already encode; (b) the
  **weak-capable share** per class — the % of instances safely routed weak, the cost-saving point
  (insight #1: always-strong is not the best policy; the gate exists only where an error cannot be
  fixed); (c) the incorrect-result percentage on τ ≥ 0.8 instances, weak path (calibration) vs
  strong path (readout) **[DECIDED, 22 Oct]**. Main metrics remain accuracy and cost; harm is
  counted, never priced (no exchange rate between wrong actions and tokens).

## 3. Models **[DECIDED, 22 Oct]**

| Role | Model | Rationale |
|---|---|---|
| Router (both stages) | Qwen2.5-1.5B-Instruct | unchanged from the certified HotpotQA campaigns; same LoRA config, same W/S logits contract |
| Strong executor | Qwen2.5-72B-Instruct | best practical open-weight model on the paper's own table (51.33 SR); our certified Config B executor family; AWQ 4-bit fits the lab card |
| Weak executor | Qwen2.5-**7B**-Instruct, first candidate | selected by the §7 pilot; fallback ladder 14B → 32B |

- **Prompts [DECIDED 22 Oct; VERIFIED 22 Oct evening]:** executor prompts = the paper's own released agent prompts, VERBATIM, for both weak and strong (reproduction-first, the AgentTTS discipline); the ONLY new prompts are the router's own instruction templates, written by us and validated line-by-line by Jonathan. Verification CLOSED: the agent prompt is in the repo (`MedAgentBench_prompt` in the task module; checked against Appendix A.2 on 23 Oct — all structural elements identical: framing, the three response formats, the placeholder set. The repo text is what their harness ran for Table 3, so it is the authoritative artifact and the one we use byte-for-byte), and the reference graders (refsol.py, Box-hosted) are downloaded, installed, and read in full.
- API models excluded: per-episode cost over ~3,300 live episodes, no weight control, no
  reproducibility guarantee mid-experiment.
- Weak-model selection rule (registered before the pilot runs): smallest ladder model whose pilot
  **query SR clears the floor** wins. **[OPEN: set the floor — starting suggestion ≥ ~30% query SR
  on the pilot sample.]** If even 32B fails, the capability cliff is itself the reported finding.
- Logistics note: 72B-AWQ (~40 GB) + 7B (~15 GB) exceed the 45.4 GB card simultaneously → episodes
  run stage-batched or with sequential model loading. **[OPEN: pick one, after the pilot.]**

## 4. Risk design **[DECIDED, 22 Oct unless marked]**

- **One registered reward function, indexed (22 Oct)**: Rᵢ = Qᵢ − λ·(Cᵢ/C_SS) − max(0, Cᵢ/C_SS − 1)
  − λt·Σₛ max(0, τᵢ − cᵢ,ₛ) — i = instance, s = stage; λ, λt, C_SS constants. τᵢ carries no stage
  subscript: one risk level per instance, applied at every stage. SCOPING: τ is computed, and c
  consulted against it, ONLY on risk-active datasets; a dormant dataset is screened by the
  small-sample τ-spread check and skips instance scoring entirely (as pW, c always exists — it is
  the routing probability itself).
- **τ per instance**, scored pre-execution by the Q1/Q2/Q3 rubric (τ = (15 − ΣQ)/12, floor 0.10,
  cap 0.95, risk floor Q1 = 1 → τ ≥ 0.8). All 300 instances scored by LLM and frozen before
  training; Jonathan validates a blind sample (3 per category, 30 total) and the agreement rate is
  reported. **[OPEN: which LLM scores — the Step-1 slide used Claude Opus 4.6.]**
- **Risk classes come from the rubric, not the HTTP verb** (22 Oct): HIGH ≡ τ ≥ 0.8 ≡ Q1 = 1;
  MEDIUM = record-mutating below the gate; LOW = read-only — recording blood pressure is a POST
  and scores τ = 0.50 (medium), so the verb alone does not decide.
- **Per-dataset activation** (21 Oct): rubric τ spread decides before training. MedAgentBench
  Step-1 spread 0.10–0.95 → **active**.
- **The risk rule — hard gate (the strong-selection rule):** τ ≥ 0.8 → route strong
  unconditionally, to avoid incorrect results that cannot be fixed; confidence is not consulted.
  0.8 is the rubric's own risk-floor condition (Q1 = 1) and is the STARTING value, refinable in
  later campaigns. Below the gate: normal routing (uniform swept θ) — where the router finds the
  weak-capable tasks.
- **cᵢ** = the router's probability that the weak choice suffices at stage i — the softmax over its
  W/S logits, the same pW the certified thresholds act on; pre-execution, zero extra cost.
  Declared campaign-2 alternative: executor-derived confidence — the weak model's log-probability on
  its PROPOSED action, checked before any POST is submitted (the verification idea in agentic form) —
  richer signal, but pays the weak pass on every instance. A **calibration audit** (reliability curve of pW vs actual weak
  success on calibration tasks, before and after training) is registered; if it fails, we report it.
- **λt:** 0 for τ < 0.8 instances (dormant); the gradient-level term is **campaign 2 only**
  (two recorded reasons: gameable own-confidence; advantage-cancellation under the mean baseline).
  If campaign 2 triggers, the wrong-action penalty P is selected by **break-even error rate**
  (−P for a wrong executed action ⇒ cheap path positive-EV only below a 1/(1+P) wrong-action rate):
  candidate set {9, 19, 99} ≙ tolerances {10%, 5%, 1%}, chosen on calibration data with a written
  rationale — a governance decision, not a valuation of harm.

## 5. Experiment design — one training run, two declared readouts **[DECIDED, 21–22 Oct]**

- **Training (once):** the validated recipe; risk term out of the gradient.
- **Readout A — without risk factor:** uniform θ from the sparse sweep (the HotpotQA protocol).
  The risk-blind, cost-optimal reference.
- **Readout B — with risk factor:** the same uniform θ **plus the τ ≥ 0.8 hard gate**.
- Both readouts come from the same run, so any difference is attributable to the risk rule alone.
- Declared reporting: accuracy @ cost for A and B; the §2 failure breakdown by risk class; the
  cost premium of B; and the calibration-measured weak wrong-action rate (§7) as the reference
  line for what the gate protects against.

## 6. Training configuration

- Trainer: **the certified grpo_trainer v1.2 file, never edited** — the MedAgentBench run imports it
  and injects a reward provider + config, so one diff shows everything that changed between
  benchmarks **[DECIDED, 21 Oct]**.
- **A1 mean-baseline advantage** (registered, twice confirmed). **G = 4** (parity shown 21 Oct;
  G = 8 fallback). λ = 0.10 carried over. Seed 17.
- **No warm start exists** (no labels). From-scratch with the validated guards (entropy floor,
  param-count check, drift check, checkpoint-by-objective). Expectation, recorded honestly: the
  HotpotQA from-scratch run landed ~1 pt under warm-started parity with the curve still rising, and
  its thresholds did not rescue runtime parity — so **[OPEN: epochs — more than 3; propose 5–6 with
  the checkpoint-by-objective rule unchanged]**.
- **Reward provider — the one new module:** lazy cache keyed by (instance, path): stored outcome if
  seen, else one live pipeline execution, stored. Valid because the machine-repeatability check (§7)
  passes first.
- **Eval frequency is now a budget line** (live greedy eval ≈ one episode per validation instance):
  **[OPEN: propose epoch-end evals on a fixed validation subsample, size set after the split
  decision in §7].**

## 7. Calibration phase (before training; all declared)

0a. **Dataset analysis (before any design freeze of the router input)** — **PRELIMINARY FINDINGS,
   22 Oct evening (v1 read in full; **v2 CONFIRMED same night**: 30 instances × 10 categories,
   exactly one instruction template per category — task1's apparent second template is an
   apostrophe-encoding difference):**
   within a category, every instruction is the SAME TEMPLATE — only the patient (MRN/name/DOB)
   varies (task 3 even records the identical "118/77 mmHg" for every patient). Consequences:
   (i) a router reading only the task text can learn exactly a 10-way template lookup — a
   per-CATEGORY W/S policy; it cannot discriminate instances, because instance difficulty lives in
   the patient's FHIR data, visible only after retrieval begins. Text-only routing degenerates to
   a fixed per-category policy; instance-level routing requires the attempt-derived view
   (insight #2 in pure form). The open-item-8 experiment stays, now testing this sharp hypothesis.
   (ii) pre-execution τ scoring reads identical text within a category → per-instance scoring will
   produce ten distinct values; the per-instance procedure stands and Jonathan's 30-instance
   validation becomes a consistency check. The gate set is in practice the high-risk categories.
   (iii) the split must stratify WITHIN categories (by patient), and the generalization claim is
   stated honestly: same templates, new patients — the router learns a per-category policy plus
   patient-data-conditioned escalation. Remaining analysis on v2: per-category instruction/context
   lengths and episode-cost profiles.
0b. **Harness reproduction check:** their prompts + their grader + Qwen2.5-72B on a declared sample; target ≈ the published 51.33 (deviations from quantization/serving recorded). A large gap = debug the harness before anything else. This measured number, not 51.33, is our always-strong ruler.
1. **Machine repeatability — SIMPLIFIED by code verification (22 Oct):** the harness NEVER executes
   POSTs against the FHIR server (payloads are accepted into the history and graded from there), and
   all ten graders issue read-only GETs — so the environment is fully static and **no reset protocol
   is needed**. The repeatability check reduces to model-output determinism (their harness runs
   temperature 0): the same ~5 tasks run twice, identical outcomes required (the 21 Oct rule:
   verify the machine is repeatable before the experiment; never change the machine mid-experiment).
2. **Model-pair pilot:** 3 tasks per category (30), run on 7B and 72B (14B/32B only if the ladder
   advances); measures per-class SR and the measured cost ratio. Selects the weak model by the §3
   rule. ~60–90 episodes.
3. **Weak-exposure measurement:** weak pipeline once on each high-risk instance (τ ≥ 0.8;
   ~60 expected) → the weak wrong-action rate e on critical instances. Safe because the FHIR server
   is simulated. ~60 episodes (overlaps partly with the pilot).
4. **τ scoring + human validation** (§4). 5. **Cost measurement** → freeze path costs and C_SS (§2).
6. **Calibration audit of pW** (§4).
- **DATA SPLIT [DECIDED 23 Oct, amended same day by team agreement]: PATIENT-DISJOINT, 150 train / 150 test**,
  frozen in `medagentbench_split_150_150.json` (supersedes the morning's 150/59/91 draw). Rationale for
  patient-disjointness unchanged: 98 patients back the 300 tasks, 89 appear in multiple categories, MRNs
  verbatim in instructions. Team agreement: a simple 150/150 top-level split (exactly 15/15 per category).
  Methodological companion: within the 150 train, exactly 6 per category (60 total) are tagged
  **train-calib** — the subset used for threshold sweeps, the model pilot, the weak-exposure measurement
  and the pW calibration audit — so no selection ever touches the test side. Construction deterministic
  (seed 20261023, randomized search + hill-climb; deviation 0 at both stages). The test split is read only
  at declared readouts.
  **[CORRECTION, 27 Oct 2026]** The 150/150 split named above (`medagentbench_split_150_150.json`, seed 20261023)
  was never materialized under that name; seed 20261023 produced only the three-way 150/60/90 draw of 23 Oct,
  retained on the lab as `medagentbench_split_20261023.json`. The split actually frozen and used by all trainings is
  the 26 Oct assignment, seed 20260729, STAIR commit 2a35c9b (`calib.txt` / `test.txt` / `calib_sweep.txt`,
  hashes in MANIFEST.json). Its generator was not preserved; its properties (150/150, 15 per type per side,
  calib_sweep 6 per type, patient-disjoint at the train/val, train/test and val/test boundaries) are verified by
  `common/tools/medagentbench_patient_split.py --check`. Provenance correction only; no design change.

## 7b. Code-verification record (22 Oct — repo, harness and graders read before any run)

- **Harness (task module, read in full):** one agent prompt (`MedAgentBench_prompt`) carrying the
  nine FHIR functions + context + question; GET requests executed live, responses injected raw;
  **POST requests are validated as JSON and "accepted" but NEVER sent to the server**; any response
  not starting GET/POST/FINISH ends the episode as an invalid action (= failure) — a strict
  format-discipline regime that explains the published small-model collapse (Mistral-7B 4%).
  `max_round = 8` confirmed in their task yaml (code default 5 is overridden). Their Qwen2.5-72B
  result was served via Together's quantized "Turbo" endpoint — so the published 51.33 is itself a
  quantized serving, making our AWQ-4bit a like-for-like deviation.
- **Graders (refsol.py, read in full):** ten per-category functions; reference answers computed
  live via read-only GETs against the static FHIR data; exact-match payload checks (NDC/SNOMED/LOINC
  codes, timestamps, units, dose formulas — e.g. task 9: dose = (3.5 − K)/0.1 × 10 mEq plus a second
  POST scheduling next-morning follow-up). Tasks 5, 9, 10 are conditional check-then-act decisions —
  exactly the instances the τ rubric scored 0.50–0.95. Query tasks fail on any POST.
- **Consequences adopted:** three-outcome mapping is mechanical (§2); no environment reset needed
  (§7.1); grading strictness noted as a reading aid for all SR numbers (their published query 38.67
  for Qwen-72B likely reflects FINISH-format exactness, not medical reading failures).
- **Pinned infrastructure (22 Oct):** serving = vLLM 0.19.0 + torch 2.10.0+cu128 in `vllm-env2`
  (newest CUDA-12 build; vLLM ≥ 0.20 requires a CUDA-13 driver the shared lab card lacks);
  harness = `mab-env` (their requirements minus the unused torch chain); FHIR server = their Docker
  image. **Operational requirement:** the 72B at full 14336 context needs the GPU exclusively
  (~44.5 GB of 46; empirically confirmed four ways against a 2.2 GB co-tenant, max feasible context
  while shared = 3,840 tokens) — runs are scheduled in exclusive windows.

## 8. Threshold sweep

Sparse pre-declared candidate grid, 8 per threshold dimension (the rehearsal passed 21 Oct with zero
loss vs the full grid), swept on calibration data; ≈ 1,500 live episodes. Uniform θ frozen before
the held-out read; Readout B applies the gate on top without re-sweeping.

## 9. Kill criteria **[OPEN — proposal]**

The HotpotQA kill baseline (both-reward pair) cannot exist here. Proposed anchors, matched currency
(delivered mean-R′ on calibration data): the run must beat **always-weak** clearly and approach
**always-strong** within a declared margin by the end of training; plus the §6 guards (lift-off from
the fresh start, no over-cheapening signature: ret P(W) and SR tracked per epoch). Exact margins to
be set once the pilot measures the weak–strong gap.

## 10. Budget summary (live episodes)

| Item | Episodes |
|---|---|
| Calibration: repeatability + pilot + weak exposure | ~150–200 |
| Training: ~150 train instances × G=4 × epochs (lazy cache reduces repeats) | ≤ ~1,800 at 3 epochs; **[OPEN: scales with the §6 epochs decision]** |
| Sweep | ~1,500 |
| Held-out readouts (A and B, declared) | ~180 |
| **Total** | **~3,600–4,500** |

## 11. Open items checklist (must close before freezing)

1. Weak-model pilot floor (§3) — suggested ≥ ~30% pilot query SR.
2. τ scores — CLOSES on Jonathan's confirmation: since every instance is its category's template,
   the Step-1 scores (Claude Opus 4.6) ARE the per-instance scores — 1: 0.10 · 2: 0.10 · 3: 0.50 ·
   4: 0.17 · 5: 0.92 · 6: 0.17 · 7: 0.13 · 8: 0.58 · 9: 0.95 · 10: 0.50, replicated ×30. Validation
   reduces from 30 instances to confirming these 10 template scores; gate set = categories 5 and 9
   (60 instances; 15/6/9 per split each, exactly on target).
3. ~~Data split~~ — DECIDED 23 Oct, amended to the team's 150/150 with train-calib tag (§7).
4. Epochs for the no-warm-start schedule (§6) — suggested 5–6.
5. Eval frequency / validation subsample size (§6).
6. Kill-anchor margins (§9) — after the pilot.
7. Serving logistics: stage-batched vs sequential loading (§3).
8. **Router input view and routing order — TBD until the dataset analysis (decided 22 Oct to defer).**
   On HotpotQA, weak-first was the solution to a measured input problem: the router could not learn
   from 100 raw passages, so the weak output became the learnable view (insight #2). Whether
   MedAgentBench's task text is learnable directly, or the router again needs an attempt-derived
   view, is an empirical question. A DATASET ANALYSIS step (added to §7) decides: what the router
   is fed, and with it the routing order — up-front W/S from the task text vs weak-first with
   escalation (record-mutating actions held until the decision). This is a DECLARED calibration
   EXPERIMENT, not a judgment call: probe which input view better predicts weak success on the
   calibration outcomes already paid for (the MedAgentBench analog of the question+100-passages
   test that produced insight #2). The same-prompts decision still
   stands: executor prompts verbatim, stage boundary = the benchmark's own interaction rounds. — **the next design discussion; nothing above depends on it except the
   final episode-count arithmetic.**

---
## OUTCOME LOG — 24 Oct: calibration steps 0b (reproduction) and 1 (repeatability) EXECUTED
- **Setup finalized 23 Oct (all recorded as local modifications):** vLLM 0.19.0 serving Qwen2.5-72B-Instruct-AWQ at `--max-model-len 11264 --gpu-memory-utilization 0.97 --enforce-eager` (0.97+14336 fails the KV-cache check — 4.38 GiB needed vs 3.54 available, engine-estimated max 11,600; 0.98 OOMs in sampler warmup; 11,264 is the stable point). FHIR store verified pre-run: 695 patients, authors' image `jyxsu6/medagentbench:latest`. Harness env additions: psutil installed; `src/client/agents/__init__.py` FastChatAgent import disabled (needs torch; our runs use HTTPAgent only); harness child processes need mab-env's bin on PATH (`python` not on system PATH). Smoke restriction (5 tasks) applied for smoke, REVERTED to byte-faithful `get_indices` before the full runs.
- **Smoke (2 × 5 tasks): PASSED, byte-identical** across runs (status, result, and every agent action). Diagnostic finding: all 5 (task1 lookups on real patients) graded incorrect because the stock FHIR server does not match a full-name string (`name=Justin Gould` → 0 results; `name=Gould` → 1): the stock prompt at temp 0 yields that query form, so this is faithful stock behavior, NOT a defect — consistent with the paper's weak query-task figure. NOTHING CHANGED in response.
- **Full reproduction, 300 tasks, run twice (23–24 Oct overnight, chained):** **run 1 = 56.67 (170/300), run 2 = 56.00 (168/300)** vs published 51.33 (154/300; Qwen2.5-72B, their Table). Same graders (their refsol, unmodified), same data (sha-pinned v2), same prompts, same image; the +~5-pt difference is attributed to the serving stack (their Together "Turbo" FP8 vs our local AWQ-4bit vLLM) — reported on slides as "reproduced in our harness," never merged with the published number. **The campaign's always-strong reference on MedAgentBench = ~56 in our harness.**
- **Status breakdown (run 1):** completedCorrect 170 · completedIncorrect 112 · task limit reached 17 · agent context limit 1 (the 11,264 cap cost at most 1 task of 300).
- **Per-category success (run 1):** task1 6.7% · task2 83.3% · task3 100% · task4 100% · task5 73.3% · task6 76.7% · task7 3.3% · task8 100% · task9 0% · task10 23.3%. Query/action mapping GRADER-DERIVED (24 Oct, from refsol.py's reject-any-POST checks): query = {task1,2,4,6,7}, action = {task3,5,8,9,10}. Our run-1 split: query 54.00 (81/150) vs their 38.67; action 59.33 (89/150) vs their 64.00 — the +5.3 overall decomposes as +15.3 query / −4.7 action: the two quantized copies have different strengths, not a uniform drift. Mechanistic reading: query tasks are graded on exact answer formatting (the paper's Fig. 2c error exhibit), and formatting habits are precisely the marginal behavior that shifts between quantizations; the paper notes their Qwen was one of only two models better at actions than queries — our copy partially reverses that anomaly.
- **Risk-design signal:** the two gated categories (τ ≥ 0.8 → strong unconditionally) split starkly — task5 73.3% but **task9 0/30 even always-strong**. The hard gate guarantees the strong path, not success; the three-outcome readout (correct / incorrect-but-fixable / incorrect-not-fixable) is what carries the safety story on task9.
- **Repeatability at scale (step 1 CLOSED):** net ±0.67 pts (170 vs 168); task-level churn 8/300 = 2.7%, confined to task5/6/7/10 (zero churn in 6 of 10 categories, including both 0% and 100% ones). Mechanism: vLLM prefix-cache/batching nondeterminism at temp 0 (smoke, cold server, was byte-identical). Adopted as the benchmark's noise floor: differences under ~±1 pt on n=300 are not signal; GRPO per-episode rewards carry ~3% outcome noise on the marginal categories.

- **24 Oct evening, REPRODUCTION VALIDATION AUDIT (triggered by Jonathan: "+5 points is too big" — every alternative explanation tested, not assumed):**
  1. **Repo deviation audit (git):** exactly three intended modifications (local vLLM agent entry in `api_agents.yaml`; assignment switch in `default.yaml`; unused FastChatAgent import disabled in `client/agents/__init__.py`) — nothing else. The smoke-edit file was restored byte-pristine via `git checkout` (the post-revert 1-line diff was only the authors' own leftover comments; executable code identical, so the runs executed exact upstream behavior).
  2. **Grader authenticity:** installed `refsol.py` sha256 `67d19da263ac92c1b7075b3218ace3a4bd19b206a06fc7aebcc897dc39625419` — MATCHES a fresh 24 Oct download from the official Stanford Box link (graders are distributed outside the repo by design; git cannot vouch for them, the hash does).
  3. **Round budget: worth 0.00 points.** All 170 successes finished in 1 tool call + FINISH (169) or 2 (1); the max_round 8-vs-5 question never comes into play. (Side-finding: successful episodes are 1–2 model calls — empirical confirmation of the one-call template analysis; feeds the router cost table.)
  4. **Decoding: identical, not a deviation.** Their committed `together-chat.yaml`: temperature 0, max_tokens 2048 — exactly our settings. Their model string: `Qwen/Qwen2.5-72B-Instruct-Turbo` (Together FP8) — the quantization difference is now documented from their own config, not inferred.
  5. **Also eliminated:** data (sha-pinned match), FHIR store (their image, 695 patients, verified pre-run), run-date drift (fixed simulated clock "2023-11-13T10:15:00+00:00" embedded in all 150 time-relative tasks), grader leniency (the smoke episodes show it grading strictly).
  6. **Significance:** pooled ours 338/600 = 56.33 vs published 154/300 = 51.33 → z = 1.42 (p ≈ 0.16): the gap is NOT statistically significant at n = 300.
  **VERDICT: reproduction faithful.** The +5 rests on documented model-quantization difference (FP8-Turbo vs AWQ-4bit — different weights arithmetic, trajectory divergence compounded over multi-round episodes) plus sampling noise, with every harness-side explanation eliminated by measurement. **Campaign baseline: always-strong ≈ 56 in our harness (170/300 and 168/300); published 51.33 reported strictly as "published (Together FP8 serving)", never merged.**
- **24 Oct, task7 failure exhibit (mechanism of the query-side gap, one episode inspected with Jonathan):** the agent answers "most recent CBG" by querying `Observation?code=GLU&patient=...&date=ge{now}` — a GREATER-OR-EQUAL filter on the current simulated time, i.e., it requests observations from the future, receives `total: 0`, and answers `[null]` (the grader computes the true latest value, with `-1` reserved for genuinely absent data). At temperature 0 this single query-construction reflex repeats across the category (task7: 1/30). CONCLUSION: between two quantized copies of the same model, query-side differences come from a handful of deterministic habits, each worth a whole 30-instance category (±20 pts of query SR) because every category shares one instruction template — the variance unit for model comparisons on this benchmark is the category, not the instance. All such failures are pure-GET (no record touched) = the incorrect-but-fixable outcome class; always-strong pays full price to fail them.
- **24 Oct late, task9 failure chain VERIFIED end-to-end (with Jonathan, episode prints + live FHIR probes):** the potassium data exists — 69 observations under base code `K`, the grader's own search key — but the agent searches with the context's LOINC `2823-3` (supplied for the ServiceRequest payload, misused as a search key), sees an empty chart, and then either loops re-querying until the 8-round limit or orders blind: 50 MedicationRequest POSTs across the category (some episodes >1, violating the grader's exactly-2-POSTs rule), only 3 ServiceRequest POSTs, dose formula unverifiable without the real K value. 0/30 on the highest-risk category (τ = 0.95, hard-gated to strong), with wrong medication orders ACCEPTED INTO THE RECORD = the incorrect-not-fixable outcome class, produced by the strong path itself. Three dead categories now each have a named, verified reflex: task1 (full-name search), task7 (ge-now on an unwindowed "most recent"), task9 (LOINC decoy). Together with the catalog's one-reflex-per-category uniformity, this is the measured mechanism behind category-level bimodality — and the strongest empirical motivation on record for the three-outcome risk readout.

- **25 Oct, ROUTER PROMPT DESIGN DECIDED (Jonathan): two registered variants.** The router prompt is the only new text written for MedAgentBench. **Variant A — "the reward in words" (general guideline, campaign-1 PRIMARY candidate):** states only the reward's active terms in plain language — answer correctly · minimal compute · correctness first, cost second · plus the risk sentence (the risk term is ACTIVE on MedAgentBench by the dataset-activation rule, unlike HotpotQA). **Variant B — objective + principles 1–2 (clear guideline, first follow-up):** everything in A plus "begin with the weak model first; escalate only when the weak output is unlikely to be correct"; principle 3 deliberately omitted (stated, it moved nothing on HotpotQA; and it is emergent from the reward, verified 24 Oct). Both texts drafted for Jonathan's verbatim validation once the MedAgentBench reward wording is final; one training run each ≈ 1,800 live episodes at G=4, so B runs only as budget allows unless promoted. **Declared decision rule:** if HotpotQA's objective-only run (launched 25 Oct 13:26) matches the stock run within seed noise, A is confirmed primary; if it falls short, B is promoted and the ladder names which stated principle earns its place. **Registered caveat on transfer:** HotpotQA's graded reward (token-F1) teaches richly, leaving the prompt little to add — MedAgentBench's binary-sparse signal may leave MORE for the prompt to do, so the HotpotQA verdict does not transfer automatically; that is why both variants stay registered.

## Serving-environment retirement — evidence record (26 Oct 2026, ~12:10 SGT)
CLAIM: The published runs' serving environment (Qwen/Qwen2.5-72B-Instruct-Turbo, FP8, Together AI) has been retired by the provider; exact reproduction is no longer possible.
EVIDENCE:
1. API error, Qwen/Qwen2.5-72B-Instruct-Turbo (request id p6jBK9w-2kFHot-a50697dc1a88f894): code "model_not_available", "Unable to access non-serverless model".
2. API error, Qwen/Qwen2.5-72B-Instruct (request id p6jBoQv-2kFHot-a5069a32cc9e3e1f): same code and message.
3. Dashboard model pages, both models: Deployment shows Serverless ✗ and Dedicated ✗ (screenshots: evidence/together_qwen72b_turbo_retired_26Oct2026.png, evidence/together_qwen72b_nonturbo_retired_26Oct2026.png).
CONSEQUENCE: the 5-point explanation closes via the best AVAILABLE test, not exact reproduction. Decided path: (i) cheap independent-serving anchor (Qwen2.5-72B-Instruct, DeepInfra via OpenRouter, quantization undisclosed) first; (ii) optional decisive FP8-vs-AWQ test (vLLM constant, rented 2xH100) depending on the anchor's result. Declared predictions to be registered per run before launch.

## Independent-serving anchor run — OUTCOME (26 Oct 2026, 13:16 SGT)
- Setup: Qwen2.5-72B-Instruct via OpenRouter, provider pinned DeepInfra (allow_fallbacks=false), temp 0, max_tokens 2048, same harness/tasks/grading as the local runs; concurrency 5; cost ≈ US$2; config: configs/assignments/openrouter_anchor.yaml, agent qwen2.5-or; outputs/DeepInfraAnchor (smoke n=5 preserved at outputs/DeepInfraAnchor_smoke).
- RESULT: success rate 58.00 (174/300 completedCorrect; 95 completedIncorrect; 31 task limit reached).
- DECLARED PREDICTION (≈56 within ±3, not ≈51): CONFIRMED — 58.00 inside the band, upper edge.
- Per-category (tasks 1–10): 20.0 / 76.7 / 100 / 100 / 80.0 / 73.3 / 6.7 / 100 / 0 / 23.3. Versus local AWQ run1 (6.7 / 83.3 / 100 / 100 / 73.3 / 76.7 / 3.3 / 100 / 0 / 23.3): identical on tasks 3,4,8,9,10; within noise on 2,5,6,7; the total gap is dominated by task 1's unstable full-name-search reflex (6.7 → 20.0), whose instability was directly observed in the smoke (same prompt, split vs unsplit name search across probes).
- VERDICT on the 5-point question: CLOSED. Two independent servings of the same weights agree (56.00/56.67 local AWQ; 58.00 DeepInfra) and sit together ~5–7 points above the published 51.33, which is one run from a provider-retired environment (see retirement evidence record). The gap is attributed to the published run's environment and/or single-run noise — demonstrated, no longer merely argued. The GRPO campaign baseline remains the local AWQ serving: 56.67 / 56.00.
- Limitation stated: DeepInfra's quantization is undisclosed — this is an independent-serving test, not a controlled FP8-vs-AWQ test (declared optional follow-up, not scheduled).

## Dataset analysis — cost accounting (item D, 26 Oct 2026)
- Method: all 300 episodes of the local always-strong run (outputs/MedAgentBenchv1) tokenized with the served model's tokenizer; per round, input = cumulative conversation (as serving bills), output = agent reply; priced with stair_common cost.units(72.0, n_in, n_out) — same function and units as HotpotQA.
- Per-category mean units (strong): t1 854.4 · t2 923.1 · t3 867.9 · t4 905.8 · t5 940.2 · t6 1028.9 · t7 885.6 · t8 898.8 · t9 4009.4 · t10 1237.1.
- C_SS (CALIB-ONLY mean, n=150, patient-disjoint split): 1236.9 units. Test episodes excluded from the constant by construction.
- FINDING: task 9 is a 3.2x cost outlier (31,081 mean input tokens from retrieval looping) AND scores 0/30 — the benchmark's strongest "escalation that cannot fix the answer" exhibit (insight 3), now with a measured price.
- REWARD-DESIGN FLAG (open, Jonathan's decision): with a global C_SS, the HotpotQA hinge max(0, C/C_SS - 1) fires at ~2.2 on any strong task-9 episode — punishing expensive INSTANCES rather than wasteful DECISIONS (on HotpotQA the hinge was provably 0). Candidates: (a) per-category C_SS; (b) per-instance C_SS_i (measured SS cost of that instance — available); (c) global C_SS with hinge dropped/retuned. Recommendation: (b).
- Status: dataset analysis items A (inventory), B (patient-disjoint split, verified OVERLAP=0 on lab), C (strong outcomes), D (costs) complete. E (weak-model outcomes), F (headroom/oracle), G (reward calibration incl. graded-Q zero-signal check) blocked on GPU availability.

## Hinge-design analysis (item D continuation, 26 Oct 2026)
- Per-instance SS cost store saved: runs/mab/ss_costs_per_instance.csv (300 rows: tid, split, n_in, n_out, units). Units min/med/max: 853 / 904 / 4,812; C_SS(calib) = 1,236.9.
- Hinge firing on PLAIN always-SS episodes (i.e., punishing baseline behavior): (a) global C_SS -> 52/300 (task9 30/30, task10 17, task6 4, task7 1; worst +2.89); per-category mean C_SS -> 157/300 (mean-split artifact; worst +0.57); (b) per-instance C_SS_i -> 0/300 by construction.
- RECOMMENDATION (pending Jonathan's decision): per-instance C_SS_i for BOTH the cost term and the hinge. Fidelity argument: on HotpotQA every question had the same SS cost, so per-instance == global there — (b) is the generalization that preserves the registered reward's meaning on a benchmark with varying instance costs, not a redesign.

## Risk-term experiment pair — DESIGN REGISTERED (26 Oct 2026, Jonathan's design)
- Two GRPO runs, one variable: R1 (no risk term): R' = Q − λ·(C_i/C_SS_i) − max(0, C_i/C_SS_i − 1). R2 (with risk): R1's reward − λ_r·max(0, τ_t − c) per stage decision. Everything else identical (fresh adapter, reward-in-words prompt per the HotpotQA ladder rule, same seed/epochs/data = calib only).
- Risk mechanism (no harm ledger exists in the benchmark): risk enters as a CONFIDENCE REQUIREMENT — stay with the cheap model only if confidence clears the task tier's bar τ_t; at deployment this is per-tier thresholds in place of one global θ. In enterprise deployment the business supplies real harm values (ATF layer); the structure is unchanged, τ/λ_r stop being proxies.
- CANDIDATE τ map (awaiting Jonathan's validation): LOW 0.50 = tasks 1,2,4,6,7 (read-only); MEDIUM 0.80 = tasks 3,8 (chart writes); HIGH 0.95 = tasks 5,9,10 (conditional/medication). λ_r: OPEN.
- PRIMARY ENDPOINT (Jonathan, 26 Oct): per-tier accuracy AT MATCHED TOTAL COST, specifically the HIGH-RISK tier τ > 0.8 (tasks 5, 9, 10). Claim form: "at equal cost, the risk-conditioned router is X points more accurate on high-risk tasks — it places escalations where errors are expensive, rather than buying more of them." Raw accuracy and cost deltas are secondary endpoints.
- DECLARED PREDICTION (three-part): (1) R2 costs more than R1 overall — by construction. (2) R2 gains accuracy on high-τ tasks only where strong > weak (sized by the pending 7B pilot). (3) Task 9 is the designed conflict: risk says escalate, insight 3 says don't (strong = 0/30 at 3.2x cost). Either outcome is a finding — the price of blanket caution, or the reward resolving a risk/futility conflict.
- Gates before launch: 7B weak pilot (items E/F), τ map + λ_r validated by Jonathan, graded-Q decision, per-instance C_SS_i adoption confirmed.

## Stage structure measured + stage-design conclusion (26 Oct 2026)
- Rounds per episode (strong model, local run 1): task1-8 = 2 (flat; task7 max 3), task10 mean 2.6 max 3, task9 mean 7.2 max 8 (the looping category). Theoretical range 1-8 (max_round=8).
- STAGE DEFINITION (Jonathan): a stage = one round (one model call) of the episode — consistent with HotpotQA (stage = one model call: retrieve, answer).
- CONCLUSION: campaign 1 trains the ROUND-1 stage decision (= per-episode routing). This is not a simplification: measured, 9 of 10 task types have exactly one substantive stage (act + FINISH), so the round-1 decision IS the stage-level decision on this benchmark. It also preserves replay economics (2 precomputable cells/instance vs path explosion under per-round switching).
- Caveat: round counts above are the strong model's. If the 7B pilot shows weak episodes running long, MID-EPISODE ESCALATION (training later-round decisions, live rollouts) is the declared follow-up chapter, justified by that measured headroom.

## RECONCILIATION with the 22 Oct registered design (26 Oct, approved by Jonathan)
1. τ VALUES: the registered Q1/Q2/Q3 rubric STANDS — per-instance scoring, (15 − ΣQ)/12, floor 0.10 / cap 0.95, all 300 frozen before training, 30 hand-validated. The 26 Oct three-tier candidate (0.50/0.80/0.95) is WITHDRAWN; "tiers" survive only as derived groupings of rubric output (high class ⇔ τ ≥ 0.8 ⇔ Q1 = 1). Jonathan's per-type confirmation is satisfied: instances of a type share task structure, so rubric scores are effectively per-type.
2. RISK EXPERIMENT SHAPE: campaign 1 reverts to the registered ONE TRAINING RUN, TWO READOUTS — risk acts at the decision layer (hard gate τ ≥ 0.8 → strong unconditionally); readout A = uniform swept θ, readout B = same θ + gate; same trained routers, difference attributes to the risk rule alone. The gradient-level risk term (−λ_r·max(0, τ − c)) is CAMPAIGN 2, as registered 22 Oct. The 26 Oct "R1/R2 two-run" entry is re-scoped accordingly.
3. KEPT from 26 Oct: the primary endpoint (high-risk-tier τ ≥ 0.8 accuracy at matched total cost), the three-part prediction (restated for readouts A/B), per-instance C_SS_i, stage = round with campaign 1 on the round-1 decision, and all dataset-analysis results.

## Items E & F CLOSED — weak pilot + headroom (26 Oct 2026, 14:45)
- Weak pilot (Qwen2.5-7B-Instruct, local vLLM at 0.50 GPU fraction, same harness/decoding): success 11.67 (35/300). Per type: t1 96.7 · t2 0 · t3 0 · t4 20.0 · t5-t10 0 (t4 aside). Statuses: 161 task-limit (8-round cap), 96 incorrect, 35 correct, 8 context-limit — weak episodes run LONG before failing (mid-episode escalation trigger exists; follow-up chapter material).
- TASK-1 REVERSAL at full scale: weak 29/30 vs strong 2/30 — the weak model searches names correctly where the strong model's full-name reflex fails. Routing gains accuracy AND saves cost on this type.
- THE BAND (vs stored strong run = run 2, 56.00; headline baseline remains 56.67 w/ run 2 as stability): always-weak 11.67 → always-strong 56.00 → ORACLE 65.00 at 52.6% of always-strong cost. Four-way: both 8 · weak-only 27 · strong-only 160 · HOPELESS 105 (35%).
- Weak mean units per type: t1 89 · t2 90 · t3 85 · t4 346 · t5 407 · t6 397 · t7 355 · t8 87 · t9 726 · t10 362 (7B tokenizer, params 7.0, same cost.units). Hopeless-set savings: e.g. task 9 weak 726 vs strong 4,009 at equal (zero) accuracy.
- Note: per-instance join and cost store derive from strong RUN 2 (outputs/MedAgentBenchv1); run 1's outputs were relocated during the reproduction work.
- DATASET ANALYSIS COMPLETE (A-F). Remaining gates to campaign 1 training: graded-Q review; τ rubric scoring protocol execution (300 scored + 30 hand-validated); G=4 and repeatability checks per the 22 Oct plan.

## Weak repeatability — lazy-cache verdict (26 Oct 2026, 16:35)
- Second full weak (7B) run: 12.00 vs run 1's 11.67 — 5/300 flips (1.7% churn), ALL in task 4, the only type where the weak model is partially competent (borderline episodes exist only there). Types at 0% and 96.7% fully stable.
- Weak churn (1.7%) < strong churn (2.7%). VERDICT: the lazy episode cache (execute each (instance, path) once, replay thereafter) stands as registered; noise floors declared: strong ±0.67 pts, weak ±0.33 pts. Flappy instances concentrate where competence is partial — noted for training-signal interpretation on task 4.
- CALIBRATION MEASUREMENTS COMPLETE. Remaining gates are all decisions: τ rubric scoring protocol (scoring LLM + Jonathan's 30-instance hand validation), graded-Q review, router-input experiment design (task text alone vs weak-attempt view, now with the weak attempt priced at 85–726 units).

## Training 1 — registered configuration (26 Oct 2026, 17:08)
- NAME: "Training 1 — risk-free training, risk tested as a guardrail at readout."
- ARCHITECTURE (Jonathan, 26 Oct): weak-first cascade — the weak model's attempt exists for every instance; the router reads the task text + a view of the weak attempt (final answer capped 400 chars, or did-not-finish + rounds; NEVER the grader's verdict) and chooses DELIVER or ESCALATE. Thesis-consistent (insight 2's input form); weak-first cost convention: deliver = C_w, escalate = C_w + C_s.
- ROUTER PROMPT (validated 26 Oct, slide "Router Prompt for MedAgentBench"): task-definition wrapper adapted to the deliver/escalate decision; objective sentence and closing VERBATIM from the HotpotQA champion; no principles.
- CORRECTNESS: pass/fail (the benchmark's own grade). Partial-credit correctness = declared follow-up check.
- REPLAY TABLE: runs/mab/training1_table.json (300 rows) from the canonical records: weak = WeakPilot7B_run1 (+/-0.33), strong = MedAgentBenchv1 (72B run 2, +/-0.67). Training is execution-free (full 2-path record).
- SPLITS: train 90 (calib minus calib_sweep) / val 60 (calib_sweep) / test 150 — patient-disjoint at every boundary.
- REWARD: R' = Q - 0.10*(C/C_SS_i) - max(0, C/C_SS_i - 1), per-instance C_SS_i = measured strong cost; weak-first path costs.
- ANCHORS (replayed): mean R' always-deliver +0.0857/+0.0914/+0.0949 (train/val/test); always-escalate +0.2091/+0.1890/+0.1640. SUCCESS BAR: the trained router must beat always-escalate's validation mean R' (+0.1890).
- WEAK-FIRST BAND: always-deliver 11.67 -> always-escalate 56.00 -> oracle 65.00 at 50.5% of always-escalate cost.
- Risk: NOT present anywhere in training; enters only as the tau >= 0.8 guardrail in the second readout tally.

## TRAINING 1 — RESULT (26 Oct 2026, 17:24; validation n=60, test SEALED)
- Run: training1_trainer v1.0, G=8, 40 epochs, seed 17, ~6 min wall; table sha 186545f0...; manifest in runs/mab/training1_router/.
- Fresh-start eval: R' 0.0104 / acc 16.7 / cost 914 / pDeliver 0.669 (untrained router: deliver-biased, far below both anchors).
- FINAL (best checkpoint): R' 0.3667 / acc 60.0 / mean cost 865.6 / pDeliver 0.425.
- VERDICT vs the registered bar (always-escalate val R' +0.1890): BEATEN by +0.178 (~1.94x) — beats_baseline true from epoch ~6 onward, plateau from ~epoch 31.
- Reading: higher accuracy than always-escalate at ~58% of its cost, learned from reward alone (pass/fail correctness, no labels, no risk). pDeliver 0.425 ~ the structural weak-favorable fraction — consistent with learning the real partition (task-1 reversal + hopeless set), to be confirmed by the per-type decision breakdown.
- Next (declared order): per-type decision analysis on val -> threshold sweep over pDeliver on val (operating points declared) -> tau rubric scoring (for the guardrail tally) -> the SINGLE held-out test read with BOTH readout tallies pre-declared.

## TRAINING 1 — VAL READOUT AND DECLARED OPERATING POINTS (26 Oct 2026, 17:30 SGT)

Per-type decision analysis (val 60, 6/type), from val_scores.jsonl joined to training1_table.json:
- DELIVER (pDeliver ~1.0): task1 (weak 6/6 — the task-1 reversal, kept), task7 and task9 (both models 0/6 — hopeless set; deliver is correct, escalation buys nothing), task10 (see below).
- ESCALATE (pDeliver ~0.0): tasks 2/3/4/5/6/8 — the strong-only territory. 9/10 types match the measured right call.
- task10 is the one miss vs val (strong 3/6 there), but on TRAIN task10 was weak 0/9, strong 1/9 — deliver was reward-optimal on the data the router saw. Data story, not a training failure. task5 mean pDeliver 0.231 with greedy 0/6: the router is not purely typing.

Threshold sweep over pDeliver on val (weak-first costs; theta=0.5 reproduces the trained result exactly):
- theta 0.00 (always-deliver): 11.7% acc, 19.3% of always-escalate cost, R' 0.0914
- theta 0.20: 56.7% acc at 51.8% cost, R' 0.3600  <- matches always-escalate accuracy at ~half its cost
- theta 0.40-0.90 (wide plateau, identical policy): 60.0% acc at 54.8% cost, R' 0.3667
- theta 1.01 (always-escalate): 56.7% acc at 100% cost, R' 0.1890

DECLARED OPERATING POINTS for the single held-out test read (before any test access):
- OP-A (primary): theta = 0.50 — the trained greedy policy.
- OP-B (cost-match): theta = 0.20 — declared claim: matches always-escalate accuracy at roughly half its cost.
Both readout tallies (without guardrail / with tau>=0.8 hard gate) will be computed for BOTH operating points from the same router pass over test 150 — replay lookups, no new episodes.

Remaining before the test read: tau rubric scoring of all 300 (scoring LLM to be named; Jonathan hand-validates 30, 3/type) — needed only for the guardrail tally.

## TAU SCORING — v1 RESULT AND THE ONE REGISTERED RUBRIC REVISION (26 Oct 2026, 18:33 SGT)
- Scorer (Jonathan's decision): the LOCAL 72B AWQ serving — same serving as the strong-model records; the Step-1 slide's scoring (Claude Opus) becomes the reference assessment, not the protocol.
- v1 run: 300/300 scored, 0 parse failures, temp 0. Within-type homogeneity excellent (8/10 types identical on every instance — confirms rubric scores are effectively per-type). RANKING matches the Step-1 assessment (tasks 9, 5 on top; read-only types at the 0.10 floor). LEVEL does not: spread 0.10-0.80, high tier (tau>=0.8) = 1 instance of 300 (one task9 Q1=1). Step-1 reference: task5 1/2/1 -> 0.92, task9 1/1/1 -> 0.95.
- Diagnosis (agreed): (a) Q3 scope ambiguity — the 72B scored reversibility of the EHR RECORD (editable) where the framework intends the CLINICAL consequence (a medication administered cannot be undone); (b) Q1 — the 72B read supplied dosing protocols as mitigation; rubric intends worst-plausible with safeguards possibly misapplied; (c) severity compression (Q=1 used twice in 900 question-scores). Old task10 "high" label WITHDRAWN on semantics: task10 orders a lab test, not a medication — the 72B's medium (0.50) stands as the better call there; the disagreement set is tasks 5 and 9 only.
- REGISTERED REVISION (one only, approved by Jonathan 18:33): two scope sentences added — Q1: score worst plausible consequence assuming task-supplied safeguards may be misapplied; a task that places or modifies a medication prescription or recommendation scores 1. Q3: score the real-world clinical consequence, not the system record. Anchors and scales UNCHANGED (Jonathan's ATF anchor tables stand). Output: tau_scores_v2.json; v1 kept on disk as the documented first pass.
- DECLARED BEFORE THE RESCORE: this is the only rubric revision; no further iteration of the judge. Jonathan hand-validates 30 (3/type, seed 20260729) against v2, agreement rate reported, and HIS scores govern any remaining disagreement. If the 72B still compresses, Jonathan's per-type scores are the frozen tau and the compression is reported as a finding (mid-size judges under-rate irreversible clinical harm).

## TAU FROZEN — v2 ACCEPTED (26 Oct 2026, 19:27 SGT)
- v2 rescore: 300/300, 0 parse failures. Tasks 5 and 9 -> Q1=1 on all 60 instances -> risk floor -> tau 0.80 HIGH (60 instances; 30 on the held-out test, 15 per type). All read-only types at the 0.10 floor; task8 0.50-0.58; task10 0.50; task3 dropped to 0.17 (v2 scope sentence reads record-only writes as reversible; below the gate either way). No overshoot. One notch below the Step-1 reference on 5/9 (0.80 vs 0.92/0.95) — same side of the gate.
- VALIDATION (Jonathan, tier level): fine Q2/Q3 values need clinical expertise a non-HCP validator does not claim; but the gate decision rides on Q1 + the risk floor, and Q1=1 for a medication order is the framework's own GOVERNANCE RULE, which Jonathan validates with full authority. Tier membership confirmed: HIGH = medication-order types (5, 9) only; medium = referral/lab-order (8, 10); low = read-only + BP recording. Q2/Q3 fine calibration recorded as LLM-scored, ordinally sensible, flagged for clinician review as future work. Validation workbook: Tau_Validation_30_26Oct2026.xlsx.
- DESIGN NOTE (Jonathan): the risk floor is the point — enterprises classify risk by policy rules on task categories (any medication order = high), not fine-grained severity estimates; hence a hard gate, not a weighted score.
- tau_scores_v2.json is FROZEN. Guardrail expectation on test, registered before the read: the gate flips only instances the router would deliver — on val the router escalates task5 already and delivers task9, so the gate's effect should be concentrated on task9: ~zero accuracy gain (strong 0/30 on task9) at added cost — prediction 3 (the task-9 conflict) playing out; task5 protected at no extra cost since the router escalates it anyway.

## TRAINING 1 — HELD-OUT TEST READ (26 Oct 2026, 19:32 SGT) — THE SINGLE READ, COMPLETE
Manifest: test_read_manifest.json; table sha 186545f0...; tau sha 2d971783...; adapter = training1_router/best_adapter.
References on test 150: always-deliver 12.00 @ 293.3 | always-escalate 54.00 @ 1,566.6 (R' 0.1640) | oracle 63.33.

OP-A (primary, theta=0.50), WITHOUT guardrail — THE HEADLINE:
  61.33% accuracy @ 847.7 mean units (54.1% of always-escalate cost), R' 0.3766, deliver 60/150.
  +7.33 accuracy points ABOVE always-escalate at 46% cost savings; within 2.0 points of the oracle.
  Val -> test generalization: val said 60.0 @ 54.8% — test 61.33 @ 54.1%. Held.
OP-A WITH guardrail (tau>=0.8 -> forced strong): 61.33% @ 1,268.7 (81.0% of SS), R' 0.3488.
  High tier (30 = tasks 5,9): acc 30.0 -> 30.0 (UNCHANGED), high-tier cost 1,029 -> 3,134 (3.0x).
  PREDICTION 3 CONFIRMED EXACTLY: the gate's only flips were task9's 15 (router delivers them; strong 0/15 there) — the insurance premium (+421 units/instance overall) bought zero accuracy on this dataset.
OP-B (cost-match, theta=0.20), WITHOUT guardrail: 60.00% @ 802.5 (51.2% of SS), R' 0.3876 (highest R' of the read), deliver 67/150; high tier 23.33 @ 803.
OP-B WITH guardrail: identical to OP-A tally2 (below-gate decision sets coincide): 61.33% @ 1,268.7.
  NUANCE: at OP-B the gate DID buy accuracy (+6.67 pp on the high tier, +1.33 overall) — the looser threshold had delivered 7 task5 instances; the gate restored them. The guardrail's value appears exactly when the cost pressure gets aggressive.
Note (declared-order discipline): OP-A remains primary as declared, although OP-B posted the higher test R' — no post-hoc switching.
VERDICT: Training 1 complete. The risk-free router beats always-escalate on BOTH axes (accuracy +7.33 pts, cost -46%); the hard guardrail is an insurance premium — free where the router already escalates (task5 at OP-A), pure cost where both models fail (task9), protective where cost pressure trades away high-risk accuracy (OP-B). This is the measured motivation for Training 2 (risk in the reward).

## CALIBRATION AUDIT (registered) + DECISION LATENCY — RESULTS (26 Oct 2026, 20:02 SGT)
Scope: val 60 (the registered calibration scope); store runs/mab/training1_calibration.json. Before = base model (a fresh LoRA's B=0 start IS the base model); after = best_adapter.
- BEFORE: probabilities smeared across the middle bins, all deliver-leaning (greedy delivers everything); ECE 0.5522. SURPRISE: AUC vs weak success already 0.9811 — the base model RANKS weak-capable instances correctly from the task text + weak-attempt view alone. Training's contribution was converting latent ranking into separated, reward-aligned decisions, not discovering the signal.
- AFTER: bimodal — 29 instances < 0.1 (weak success 3.4% there), 24 > 0.9 (holds 6 of the 7 weak-correct val instances; empirical weak success 0.25 because the top bin ALSO holds the hopeless deliveries, which are reward-correct). Saturation: 83.3% of probabilities outside [0.05, 0.95] — the quantified "bunched probabilities, few operating points" finding behind the flat theta plateau (0.40-0.90).
- INTERPRETATION (for the metrics slide): as a LITERAL probability, pDeliver fails calibration (ECE 0.3355; 0.99 does not mean 99% weak success) — by design, since the trained target is the reward-optimal ACTION (delivering hopeless instances is correct and cheap). As a DECISION SCORE it is strong: AUC 0.9326 vs weak success even though that ground truth counts every correct hopeless-deliver against it.
- DECISION LATENCY: mean 19.1 ms / median 18.8 / p95 24.5 per routing decision (one 1.5B forward pass, model resident, n=60). Decision overhead is negligible in time as well as compute (episodes run seconds to minutes).
- GRADER PROVENANCE VERIFIED (19:58): upstream eval.py (unmodified, stanfordmlgroup/MedAgentBench @ 9926011) is a dispatcher that imports the separately-distributed refsol.py by the authors' own design ("Make sure to download the refsol.py..."); our only repo changes are client/config-side (one torch-dependent import commented out; agent/assignment configs). completedCorrect is written by their grading path — our accuracy IS the published success-rate metric. OPEN: one line documenting where our refsol.py copy was downloaded from (Jonathan to confirm source).

## PARTIAL-CREDIT CHECK — DEFERRED WITH RATIONALE (26 Oct 2026, 20:26 SGT, Jonathan's call)
- The declared trigger ("after Training 1, if its learning looks starved or on Jonathan's call") is NOT met: Training 1 converged cleanly (6 min), beat its bar 1.9x, and landed within 2.0 points of the oracle on the held-out test — no starvation to remedy.
- Design work done and kept on file: the weighted-components proposal (withdrawn as impractical — ~20 hand-chosen weights) and the approved MECHANICAL scheme (graded Q = fraction of the official grader's own assertions passed; safety gate -> 0; branch logic preserved; official pass => 1.0 — zero free parameters). If the check is ever run, the mechanical scheme is the registered form.
- Known effect if run: the only decision graded Q could plausibly flip is task-9-style (escalate where strong scores partial credit but zero pass/fail) — which would RAISE cost with ZERO gain on the benchmark's own pass/fail metric. Recorded as the reason the check is unlikely to help this dataset.
- Status on the results board: "deferred — not needed (learning not starved; near-oracle)".

## TRAINING 2 — REGISTERED CONFIGURATION (26 Oct 2026, 20:30 SGT, before any run)
- NAME: "Training 2 — risk in the reward." QUESTION: can a router that LEARNS risk protect high-risk tasks more cheaply than the blanket guardrail (whose measured price is +27 cost points for +0.0 accuracy at OP-A)?
- REWARD (the full registered master form, now with the last term active): R'2 = Q(pass/fail) - 0.10*(C/C_SS_i) - hinge - lambda_t*max(0, tau - c), DELIVER branch only, dormant on tau < 0.8. tau = frozen tau_scores_v2.json (sha in manifest); c = the router's OWN deliver-probability at decision time, DETACHED (prices the decision; no gradient path through c). The sum over stages collapses to the single round-1 decision (stage = round).
- EVERYTHING ELSE IDENTICAL to Training 1 (same table, splits, prompt, GRPO mechanics, fresh adapter, seed 17) — one variable changes, so differences attribute to the risk term alone.
- LAMBDA_T: declared candidate set {0.5, 1.0, 2.0}; selected on VALIDATION mean delivered R'2; selection rule declared here, before any run. Checkpointing is by R'2; the base R'1 is reported alongside for comparability.
- CODE: training2_trainer.py v1.0, generated as a mechanical delta on the validated training1_trainer.py v1.0 (unchanged lines are the audited lines); pending Jonathan's line-by-line validation.
- REGISTERED PREDICTIONS (before any run):
  P1. With saturated confidence, the penalty binds asymmetrically: on task 5 (router already escalates, pW~0) it adds pressure in the direction already taken; on task 9 (router delivers at pW~1) max(0, 0.8 - 1) = 0 — the penalty is SILENT, so the router may legitimately keep delivering task 9. If so, Training 2 achieves task-5 protection at zero extra cost AND avoids the guardrail's task-9 premium — beating the gate.
  P2. The gameability risk (recorded 22 Oct): the penalty shrinks as confidence RISES, so a perverse route to low penalty is overconfident delivery on high-risk instances. Watch: high-tier pDeliver drifting UP during training without outcome support. If observed, reported as the registered failure mode.
  P3. Early-training effect: while probabilities are mid-range, the penalty actively steers high-tau instances toward escalation — expected to show as faster high-tier separation than Training 1's curve.
- EVALUATION: against Training 1's readouts on VALIDATION (incl. high-tier deliver/acc/cost vs the guardrail tallies). A second held-out test read ONLY if declared separately after a material validation result. The test is not a tuning set.

## TRAINING 2 — RESULT (26 Oct 2026, 21:00 SGT) — CONCLUDED ON VALIDATION
Three runs, lambda_t in {0.5, 1.0, 2.0}, everything else identical to Training 1 (manifests in runs/mab/training2_router_lt*/; logs training2_lt*.log; fresh-start base R' 0.0104 reproduced Training 1's exactly).
- OUTCOME: all three lambda_t converged to the IDENTICAL greedy policy — deliver task 1 only, escalate everything else. val: acc 65.0, cost 1,494.3, R'2 = R'1 = 0.2927, high tier 0/12 delivered, acc 0.333 @ 3,172.6. Lambda_t selection: three-way tie, recorded as such (result is lambda_t-insensitive across the declared set).
- THE EXACT VAL COMPARISON: T1 greedy 60.0 @ 865.6, R' 0.3667 | T1+guardrail 60.0 @ 1,282.0, R' 0.3348, high tier 0/12 @ 3,172.6 | T2 65.0 @ 1,494.3, R' 0.2927, high tier 0/12 @ 3,172.6.
- PREDICTIONS RESOLVED: P1 REFUTED (no task-9 differential; high-tier outcome numerically identical to the hard gate). P3 CONFIRMED (penalty swept the high tier to escalation by ~epoch 3 while confidence was mid-range; low-confidence/high-penalty loop locked it for the rest of training, at every lambda_t). P2 NOT OBSERVED (no confidence-inflation gaming; the failure mode was lock-in).
- T2's +5 accuracy is INCIDENTAL: it comes entirely from escalating task 10 — a BELOW-GATE type (tau 0.50) the risk term does not govern — via spillover (the penalty's pressure left the router escalate-happy; it never learned to deliver the hopeless types, cost 1,494 vs 866). The same accuracy is available cheaper from T1-based policies (T1+gate+escalate-task10 ~ 65.0 @ ~1,406). Val-noise caveat: task 10 strong is 3/6 on val vs ~23% overall.
- SECOND TEST READ: NOT TRIGGERED — the declared condition (material validation win over the guardrail) is not met. Training 2 concludes on validation; the held-out test remains read once.
- VERDICT: the gradient-level risk term reproduces the guardrail's protection at identical high-tier numbers while costing more everywhere else (shared-weight spillover degrades the cost-accuracy policy). RISK BELONGS AT THE DECISION LAYER, NOT IN THE GRADIENT. Training 1 + readout gate is the kept architecture.
- IMPROVEMENT PATH NOTED (found independently by the guardrail readout and T2): Training 1's one weak delivery type is task 10; escalating it by per-type policy is a cheap, policy-level fix to review before any deployment claim — separate from the risk question.

## REFSOL PROVENANCE — CLOSED (26 Oct 2026, 21:03 SGT)
- refsol.py (the per-task grading functions) obtained from the download link in the official MedAgentBench README (stanfordmlgroup/MedAgentBench) at benchmark setup, October 2026 — the authors' own distribution channel; their shipped eval.py dispatcher instructs exactly this installation.
- Content verified identical between the lab installation (src/server/tasks/medagentbench/refsol.py) and Jonathan's archived copy (C:\Documents\Research\slr), 26 Oct. Grader-provenance chain fully documented: unmodified upstream dispatcher -> officially distributed refsol -> completedCorrect -> our accuracy numbers.

## TRAINING 2 — CALIBRATION AUDIT (26 Oct 2026, 22:11 SGT)
Same audit as Training 1's, lt0.5 adapter (tie broken by smallest lambda_t; all three converged to one policy). Store: runs/mab/training2_calibration.json. Before-training pass reproduced the base-model numbers exactly (built-in consistency check).
- After training: AUC 0.9987, ECE 0.0318, saturation 76.7%, latency 18.8 ms (identical architecture).
- THE CHARACTERIZATION THIS COMPLETES: Training 1's ECE 0.3355 came from confidently delivering the HOPELESS instances (reward-correct, literally miscalibrated); Training 2 escalates those, so its deliver-confidence is high only where the weak model actually succeeds (top occupied bin: 6 instances, mean 0.798, empirical weak success 1.00 — task 1). The risk term BOUGHT LITERAL CALIBRATION AND SOLD COST-EFFICIENCY: T2's pDeliver ~ honest P(weak succeeds); T1's pDeliver = an action score. Same protection, same latency; the choice between them is cost (T1 wins by 40 points) vs literal probability honesty (T2) — and the decision-layer architecture keeps T1's cost while treating its score as a score.

## MID-EPISODE ESCALATION — STEP A: HEADROOM MEASURED (26 Oct 2026, 22:19 SGT, replay-only)
- Store: runs/mab/weak_costs_by_round.csv (cumulative weak cost per round, all 300; same tokenization as the cost store). Re-priced: the Training 1 OP-A decisions on val+test (210), held fixed — no new outcomes touched; the held-out test stays read once.
- HEADROOM (upper bound, decisions held fixed): escalate at round 1 -> 46.9% of always-escalate (7.4pp saved); round 2 -> 48.6% (5.7pp); round 3 -> 49.5% (4.8pp). Current architecture: 54.3%.
- WHERE THE WASTE LIVES: escalated tasks 4/5/6 burn 7-8 rounds (~349/408/398 units) before failing; escalated tasks 2/3/8 stop at round 2 (~86-90 units, nothing to save).
- FEASIBILITY NOTE: Training 1's decisions are type-level, and type is recognizable at round 2 (arguably from task text alone) — so the round-2 bound is likely realizable, not just theoretical. The 5.7pp is also THE MEASURED PRICE OF THE WEAK-FIRST CASCADE (the cost of always letting the weak attempt finish).
- DECLARED NEXT (pending Jonathan's go): "Training 3" — the round-2 router: identical replay machinery, view = the first two rounds of the weak attempt (constructible from stored transcripts), train/val as before; prize = 48.6% of always-escalate at unchanged accuracy if decisions are preserved; if the partial view degrades decisions, that is the honest limit of early escalation. Evaluation on validation; any test read to be declared separately.

## TRAINING 3 — REGISTERED CONFIGURATION (26 Oct 2026, 22:21 SGT, Jonathan: "run rather than assume")
- NAME: "Training 3 — the round-2 router." QUESTION: does the round-2 partial view preserve Training 1's decisions, realizing the Step A headroom (48.6% of always-escalate at unchanged accuracy), or does early information degrade them (the honest limit of early escalation)?
- DESIGN: identical to Training 1 (risk-free; guardrail stays a readout tally) except the VIEW (weak attempt cut at round 2: <=2-round episodes keep the T1 summary; longer ones show "still working" + the round-2 action) and the ESCALATE cost (cw2 + cs; deliver still pays full cw, since delivering requires the finished attempt). Table: training3_table.json w/ new cw2 field. Same splits, prompt (verbatim — "has already attempted it" reads loosely mid-attempt; kept fixed for comparability), GRPO mechanics, seed.
- CODE: training3_build_table.py + training3_trainer.py, mechanical deltas on the validated T1 pair; pending Jonathan's checkpoint validation.
- PREDICTION (before the run): decisions are type-level and type is visible at round 2, so the router should reproduce T1's partition (deliver task1 + hopeless, escalate the strong-only types) and land near 60% val accuracy at ~49% of always-escalate cost; the risk is confusion among still-working episodes (e.g., task5 vs task9 both unfinished at round 2) — a drop there would show as high-tier or hopeless-set decision flips.
- EVALUATION: validation vs Training 1; kill bar = always-escalate val mean R' under round-2 accounting (printed by the table builder). Any test read to be declared separately; the held-out test stays read once.

## TRAINING 3 — RESULT (26 Oct 2026, 22:40 SGT) — CONCLUDED ON VALIDATION
Run: training3_router (log training3_full.log; table sha 4eff23f3...; bar = always-escalate-at-round-2 val R' +0.3719).
- RESULT: val R' 0.4756 (beats the bar by +0.104), acc 65.0, mean cost 1,278.2, pDeliver 0.142; converged by epoch ~3, stable to epoch 40. Fresh start 0.1234/20.0.
- PER-TYPE (val): deliver task1 ONLY (6/6, pD 0.959); ALL other types escalated (pD <= 0.32 everywhere, mostly <0.07) — hopeless types and task10 included.
- PREDICTION RESOLVED AGAINST ITSELF, INSTRUCTIVELY: the router did NOT reproduce Training 1's decisions at lower cost (the Step A bound). Cheap round-2 escalation CHANGED THE OPTIMAL POLICY: hopeless-delivery's cost advantage shrank from ~3,300 units to the round-2 residue, and escalating task10 became reward-positive — so the reward's optimum moved to "deliver only what the weak model has already finished correctly by round 2; escalate everything still working." Val-oracle accuracy (65.0) at 1,278.
- THE LEGIBLE RULE (governance note): the round-2 view makes the policy operationally explainable — task1's weak episodes FINISH within 2 rounds (the router sees a completed answer); every still-working episode escalates immediately instead of burning 6 more rounds.
- DOMINANCE: Training 3 strictly dominates Training 2 at the 65% accuracy point — same accuracy, 1,278 vs 1,494 (-14%), no risk term, high tier fully escalated (guardrail satisfied for free). The high-accuracy corner of the board belongs to the round-2 architecture.
- THE FRONTIER NOW HAS TWO TRAINED POINTS: Training 1 (60.0 @ 865.6 — the cost-efficient point; R' 0.3667 under full-episode accounting) and Training 3 (65.0 @ 1,278.2 — the accuracy point; R' 0.4756 under round-2 accounting). Accountings differ (escalation pricing), so R' values are not directly comparable across the two; costs and accuracies are.
- EVALUATION: concluded on validation per the registration; any test read for Training 3 to be declared separately — the held-out test stays read once.

## TRAINING 3 — HELD-OUT TEST READ DECLARED (26 Oct 2026, 22:43 SGT, Jonathan's go: "do all three")
- SCOPE: ONE router pass of the Training 3 best adapter over the test 150 (round-2 views from training3_table.json). This is Training 3's first and only test read; Training 1's read stands untouched; each trained router gets exactly one read.
- DECLARED OPERATING POINT: theta = 0.50 only (the trained greedy policy; T3's probabilities are saturated and no cost-match secondary point was swept or declared). Both readout tallies reported as registered (without guardrail / with tau >= 0.8 forced strong) — expected to coincide, since the val policy escalates the entire high tier; reported regardless.
- COSTS: Training 3's true deployment economics (deliver = full cw; escalate = cw2 + cs). Reported two ways: % of always-escalate under round-2 accounting (its own economy) AND % of the board's full-episode always-escalate (cross-row comparability).
- PREDICTION (before the read): the policy is type-level and saturated, so test should land near val: ~63-65% accuracy (test task10 strong is weaker than val's 3/6 — the main downside risk), at ~78-82% of full-episode always-escalate cost.

## TRAINING 3 — CALIBRATION AUDIT + GITHUB SNAPSHOT (26 Oct 2026, 22:44 SGT)
- T3 calibration (round-2 views; store runs/mab/training3_calibration.json): after training AUC 0.907, ECE 0.0396, saturation 80%, latency 19.7 ms. Top bin = task1 only (6 @ mean 0.959, empirical weak success 1.00). THE THREE-POINT PATTERN: both escalate-the-unfinished policies (T2, T3) produce literally-calibrated confidence (ECE 0.03-0.04); Training 1's confident hopeless-delivery makes its confidence an action score (ECE 0.34) — same AUC class in all three.
- GITHUB SNAPSHOT: branch jonathan @ 446c3d4 — lab_package/medagentbench/ with all ten campaign scripts + this registration + README. No episode stores, adapters, or credentials committed.
- MID-EPISODE LIVE-ROLLOUT VARIANT (bucket-2b) SCOPED AS THE NEXT CHAPTER, not runnable from replay: strong CONTINUES from the weak model's partial state (handoff) instead of restarting — needs new live episodes (both models served), a handoff protocol, and its own cost accounting; registered as future work with the round-2 restart results (T3) as its baseline.

## TRAINING 3 — CALIBRATION AUDIT (26 Oct 2026, 22:46 SGT)
Store: runs/mab/training3_calibration.json (round-2 views; before = base model on the same views).
- After training: ECE 0.0396, AUC 0.907, saturation 80%, latency ~20 ms. Top bin = 6 @ 0.959 with empirical weak success 1.00 (task1's finished answers); the stray weak-correct task4 instance sits escalated in the low bin (the AUC cost).
- Pattern across the three routers: T2 and T3 (escalate-everything-uncertain) both achieve honest literal probabilities (ECE 0.03-0.04); T1 alone trades literal calibration (ECE 0.34) for cost-efficiency via confident hopeless deliveries. Calibration honesty and cost-efficiency trade off through the hopeless set.
- Note: the base model under ROUND-2 views ranks worse than under full views (before-AUC 0.908 vs 0.981) — the finished attempt carries ranking signal the partial view lacks; training recovers the decisions type-level anyway.

## FUTURE CHAPTER SCOPED — MID-EPISODE HANDOFF (live rollouts) (26 Oct 2026, 22:47 SGT)
- QUESTION: does CONTINUING the strong model from the weak model's partial state (handoff) beat restarting it (our round-2 architecture), net of the contaminated-context risk?
- WHY IT CANNOT BE REPLAYED: every stored strong episode starts from scratch; a handoff's trajectory depends on the weak prefix, so outcomes require NEW live episodes (both models served, harness support for mid-episode context transfer). Cost class: real episodes, not lookups.
- WHAT IS ALREADY IN HAND: the per-round cost store (weak_costs_by_round.csv), the round-2 decision architecture (Training 3), and the measured restart baseline it must beat (65.0 @ 1,278 val).
- STATUS: scoped, not scheduled. Any execution begins with its own registration (design, budget, predictions).

## TRAINING 3 — HELD-OUT TEST READ (26 Oct 2026, 22:50 SGT) — ONE READ, COMPLETE
Manifest: training3_router/test_read_manifest.json; table sha 4eff23f3...; tau sha 2d971783... (version string cosmetically says training1_test_read — the delta missed the VERSION line; paths/hashes identify the run).
- RESULT (OP-A theta 0.50, the only declared point): accuracy 63.33 = THE TEST ORACLE, at 1,276.3 mean units = 81.5% of full-episode always-escalate (1,566.6) and 93.7% of round-2 always-escalate (1,361.7). R' 0.4588 vs bar 0.345. Deliver 15/150 (task1; all correct).
- Both tallies identical (high tier fully escalated) — the guardrail is satisfied at zero extra cost, as predicted.
- PREDICTION HELD: 63.33 in the declared 63-65 band, at the bottom edge for the foreseen reason (test task10 harder than val's 3/6; escalation absorbed it).
- THE FINAL FRONTIER (held-out test, both routers read once each): Training 1 — 61.33 @ 54.1% of always-escalate (the cost point); Training 3 — 63.33 (oracle) @ 81.5% (the accuracy point). Training 2 remains validation-only and dominated. CAMPAIGN COMPLETE.

## PROTOCOL AMENDMENT — COMPLETENESS READ FOR TRAINING 2 (26 Oct 2026, 22:58 SGT, approved by Jonathan)
- AMENDMENT: every trained router is reported on the held-out test exactly once (apples-to-apples board). Training 2's read is added for comparability only — its policy has been frozen since the 21:00 result (lambda_t three-way tie, identical policies; lt0.5 adapter read; no selection is possible), and the original condition ("material validation win") remains the rule for any PROMOTION decision. This read reports, it does not promote.
- SETUP: Training 1's table (full-attempt views = what T2 was trained on; full-episode escalation costs = T2's real economics), single declared point theta = 0.50, both tallies, run-once guard in training2_router_lt0.5/.
- PREDICTION (before the read): T2's decisions = deliver task1 only (the same decision set as Training 3), so test accuracy should land at exactly 63.33 (the oracle), at ~94-95% of always-escalate cost — placing the T3-dominates-T2 claim on the held-out test itself (same accuracy, ~13 points more cost).

## TRAINING 2 — COMPLETENESS READ RESULT (26 Oct 2026, 23:02 SGT) — PREDICTION EXACT
- Manifest: training2_router_lt0.5/test_read_manifest.json (training2_test_read v1.0; T1 table sha 186545f0...).
- RESULT: 63.33 accuracy @ 1,481.2 mean units = 94.5% of always-escalate; R' 0.2778; deliver 15/150 (task1); both tallies identical (high tier fully escalated). The registered prediction ("exactly 63.33 at ~94-95%") landed exactly.
- THE BOARD IS NOW APPLES-TO-APPLES: all three trained routers read once each on the held-out test. T1: 61.33 @ 54.1% (cost point). T2: 63.33 @ 94.5%. T3: 63.33 @ 81.5% (accuracy point). THE DOMINANCE IS A TEST FACT: T3 = T2 accuracy at 13.0 points less cost. Verdict unchanged and now test-grounded: early escalation, not a risk term, wins the accuracy corner; risk belongs at the decision layer.

## WEAK-EXPOSURE MEASUREMENT — §7 STEP 3 EXECUTED (28 Oct 2026, 17:07 SGT) — e = 0 ON THE HIGH TIER
- Purpose: decide whether an action-based risk term, −P·1[τ≥0.8]·1[deliver]·1[weak attempt executed a wrong action] with P=(1−e*)/e* (tolerance e* ∈ {10%, 5%, 1%} ⇒ P ∈ {9, 19, 99}; the gate is the P→∞ limit), can replace the confidence proxy −λt·max(0, τ−c) of Training 2. Prerequisite: the weak wrong-action rate e on the high tier.
- Method: read-only replay over the canonical weak store (WeakPilot7B_run1/qwen2.5-7b-local/medagentbench-std/overall.json, sha256 db008f80…), τ from results/medagentbench/tau_scores_v2.json (2d971783…, frozen), the committed splits. Script lab_package/medagentbench/weak_exposure_measure.py; report results/medagentbench/weak_exposure_high_tier.json (per-instance flags). "Accepted POST" = refsol.extract_posts' definition (a POST whose next message contains "POST request accepted"); wrong_order = graded incorrect AND ≥1 accepted MedicationRequest POST. No model run, no FHIR server.
- RESULT (all 60 τ≥0.8 instances = tasks 5 and 9; train 18 / val 12 / test 30): correct 0/60. Task 5: no_action 30/30 — every episode "task limit reached" at 8 rounds with 0 POSTs (omission by exhausting the round budget on lookups). Task 9: rejected_post 30/30 — every episode attempts 2–8 POSTs (189 in total), every one answered "Error in sending the GET request: 400" (the GET+POST bundle is parsed as a GET; verified by listing every harness reply after a POST — one distinct string); statuses: task limit reached 18, agent context limit 11, completedIncorrect 1. **e_accepted = 0.000** on every split; **e_attempted = 0.000 (task 5), 1.000 (task 9)**.
- DECISION: the action-penalty training ("Training 4") is NOT run — the penalty term is identically zero on the training set, so the recipe would reproduce Training 1. No GPU spent.
- INTERPRETATION: the high tier is 30.0 % under every policy because the weak model scores 0/60 there and the router's choice never changes a high-tier outcome; the Training 2 term had no harm to price and priced the router's own confidence instead (hence λt-insensitivity; its +2 points came from task 10). Design rule: measure e before putting a risk term in the reward; with e = 0 the right policy is cost-only routing plus capability, and the gate's +27 cost points guard against a hazard that does not occur in this harness. Caveat for deployment: task 9's safety is the harness format check, not the model (intent-to-act 100 %, correctness 0 %); e_accepted is harness-dependent and could reach 1.0 under a lenient harness — the argument for a decision-layer guardrail, and the honest qualifier on "the risk term is not needed *given this harness*".

