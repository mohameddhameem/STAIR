# router_prompts_objective.py -- reward-objective-only prompt variant (v1.0, 25 Oct 2026)
# The system prompt states ONLY the reward's objective (correctness first, cost
# second) -- no procedural principles: no weak-first framing, no escalation rule,
# no cheap/expensive descriptors. Stage instructions are untouched (they define
# what the options mean -- the task, not advice).
import router_prompts as _RP

OBJECTIVE_TEMPLATE = (
    "You route a question-answering pipeline. Each stage runs either {weak_name} "
    "or {strong_name}. Your objective: the pipeline should answer correctly while "
    "spending as little compute as possible - correctness first, cost second. "
    "Based on the information shown below, choose one option."
)

def activate():
    _RP.SYS_CASCADE_TEMPLATE = OBJECTIVE_TEMPLATE

def _selfcheck():
    assert "{weak_name}" in OBJECTIVE_TEMPLATE and "{strong_name}" in OBJECTIVE_TEMPLATE
    for banned in ("weak model first", "escalat", "cheap)", "expensive"):
        assert banned not in OBJECTIVE_TEMPLATE, f"procedural wording found: {banned}"
    OBJECTIVE_TEMPLATE.format(weak_name="a", strong_name="b")
    print("objective prompt OK -- reward objective only, no procedural principles, format fields valid")

if __name__ == "__main__":
    _selfcheck()
