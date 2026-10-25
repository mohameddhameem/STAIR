# router_prompts_guided.py -- F6g prompt variant (v1.0, 25 Oct 2026)
# The stock system prompt already states insights 1-2 (weak-first; escalate only
# when needed; cheapest-correct). This module appends the missing THIRD insight.
# Nothing else changes: all templates, assemble(), letter_ids remain the stock
# module's own, so the only delta between F6 and F6g is GUIDANCE below.
import router_prompts as _RP

GUIDANCE = ("Routing principles: (1) Begin with the weak model first. "
            "(2) Escalate to the strong model only when the weak output is unlikely to be correct. "
            "(3) If even the strong model would be unlikely to answer correctly, keep the weak "
            "choice - escalation that cannot fix the answer only adds cost.")

_STOCK_TEMPLATE = _RP.SYS_CASCADE_TEMPLATE
GUIDED_TEMPLATE = _STOCK_TEMPLATE + " " + GUIDANCE

def activate():
    """Every caller of the stock module then sees the guided system prompt."""
    _RP.SYS_CASCADE_TEMPLATE = GUIDED_TEMPLATE

def _selfcheck():
    assert GUIDED_TEMPLATE == _STOCK_TEMPLATE + " " + GUIDANCE
    assert GUIDED_TEMPLATE.startswith(_STOCK_TEMPLATE) and GUIDED_TEMPLATE.endswith(GUIDANCE)
    print("guided prompts OK -- stock template + guidance, byte-exact")

if __name__ == "__main__":
    _selfcheck()
