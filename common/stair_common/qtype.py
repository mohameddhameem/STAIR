"""
stair_common/qtype.py -- question TYPE from the wording, one rule set per dataset. v1.0 2026-11-11 (DRAFT for Jonathan's review).
Purpose: a router input line "Question type (from wording): <type>" that is IDENTICAL at training, validation and test, on
every dataset, and never reads a dataset's own type field (which is unavailable at deployment).
  2wiki     -> comparison | bridge-comparison | compositional | inference   (the 10 Nov rule of twowiki_type_routing.py,
               verbatim; 94.2 % agreement with 2Wiki's gold type on val400; written before its accuracy was seen)
  hotpotqa  -> comparison | bridge                                          (HotpotQA has two types; the same comparison
               keywords; everything else is bridge)
This module is loaded by FILE PATH (importlib) on the lab's Python 3.10, like score.py and cost.py; it has no imports beyond re.
TYPE_LINE is the exact text appended to the router's user message; the test store must append the same line.
Self-check: python3 -I qtype.py
"""
import re

VERSION = "qtype v1.0 2026-11-11"
TYPE_LINE = "\nQuestion type (from wording): {qtype}."

KIN = re.compile(r"\b(grandfather|grandmother|grandparent|grandchild|grandson|granddaughter|uncle|aunt|nephew|niece|"
                 r"father-in-law|mother-in-law|son-in-law|daughter-in-law|brother-in-law|sister-in-law|"
                 r"stepfather|stepmother|step-father|step-mother|paternal|maternal)\b", re.I)
CMP = re.compile(r"(\bor\b|\bboth\b|\bsame\b|\bfirst\b|\bearlier\b|\blater\b|\bolder\b|\byounger\b|\blonger\b|"
                 r"\bmore recently\b|\bfirst\b|\blast\b|\bborn first\b|\bdied first\b|\bdied later\b)", re.I)
ROLE = re.compile(r"\b(director|directed|composer|performer|producer|creator|author|writer|founder|editor)\b", re.I)

TYPES = {"2wiki": ("comparison", "bridge-comparison", "compositional", "inference"),
         "hotpotqa": ("comparison", "bridge")}


def question_type(dataset, question):
    d = dataset.lower()
    if d == "2wiki":
        if KIN.search(question):
            return "inference"
        if CMP.search(question):
            return "bridge-comparison" if ROLE.search(question) else "comparison"
        return "compositional"
    if d == "hotpotqa":
        return "comparison" if CMP.search(question) else "bridge"
    raise ValueError(f"no type rule for dataset {dataset!r}")


def type_line(dataset, question):
    return TYPE_LINE.format(qtype=question_type(dataset, question))


def _selfcheck():
    assert question_type("2wiki", "Who is the maternal grandfather of Ptolemy IX?") == "inference"
    assert question_type("2wiki", "Which film was released first, A or B?") == "comparison"
    assert question_type("2wiki", "Are the director of A and the director of B from the same country?") == "bridge-comparison"
    assert question_type("2wiki", "Where was the place of death of the composer of X?") == "compositional"
    assert question_type("hotpotqa", "Which magazine was started first, Arthur's Magazine or First for Women?") == "comparison"
    assert question_type("hotpotqa", "The Oberoi family is part of a hotel company that has a head office in what city?") == "bridge"
    assert type_line("2wiki", "Who is the uncle of X?") == "\nQuestion type (from wording): inference."
    print(VERSION, "self-check OK")


if __name__ == "__main__":
    _selfcheck()
