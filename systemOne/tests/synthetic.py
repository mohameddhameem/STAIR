"""Synthetic HotpotQA-shaped examples, for running the pipeline without downloads."""

import random

import pandas as pd

from stair.prepare_hotpotqa import flatten_example

SYLLABLES = "ka lo mer vin sa tor el ban ri qua dos fen lu mar ot is ren gal".split()


def name(rng: random.Random, words: int = 2) -> str:
    return " ".join("".join(rng.sample(SYLLABLES, 2)).capitalize() for _ in range(words))


def film_facts(rng: random.Random) -> dict:
    return {"film": name(rng), "director": name(rng), "city": name(rng, 1), "year": rng.randint(1950, 2020)}


def film_paragraphs(f: dict) -> list[tuple[str, list[str]]]:
    return [
        (f["film"], [f"{f['film']} is a {f['year']} film directed by {f['director']}.", "It was well received."]),
        (f["director"], [f"{f['director']} is a filmmaker.", f"{f['director']} was born in {f['city']}."]),
    ]


def make_example(rng: random.Random, i: int, level: str) -> dict:
    a, b = film_facts(rng), film_facts(rng)
    kind = rng.choice(["bridge", "comparison_yesno", "comparison_which"])
    if kind == "bridge":
        question = f"In which city was the director of {a['film']} born?"
        answer, qtype = a["city"], "bridge"
        gold = film_paragraphs(a)
        sp = [(a["film"], 0), (a["director"], 1)]
    elif kind == "comparison_yesno":
        same = rng.random() < 0.5
        if same:
            b["year"] = a["year"]
        question = f"Were {a['film']} and {b['film']} released in the same year?"
        answer, qtype = ("yes" if same else "no"), "comparison"
        gold = film_paragraphs(a)[:1] + film_paragraphs(b)[:1]
        sp = [(a["film"], 0), (b["film"], 0)]
    else:
        question = f"Which film came out first, {a['film']} or {b['film']}?"
        answer = a["film"] if a["year"] <= b["year"] else b["film"]
        qtype = "comparison"
        gold = film_paragraphs(a)[:1] + film_paragraphs(b)[:1]
        sp = [(a["film"], 0), (b["film"], 0)]

    distractors = []
    while len(gold) + len(distractors) < 10:
        distractors += film_paragraphs(film_facts(rng))
    paragraphs = gold + distractors[: 10 - len(gold)]
    rng.shuffle(paragraphs)
    return {
        "id": f"syn{i:05d}",
        "question": question,
        "answer": answer,
        "type": qtype,
        "level": level,
        "context": {"title": [t for t, _ in paragraphs], "sentences": [s for _, s in paragraphs]},
        "supporting_facts": {"title": [t for t, _ in sp], "sent_id": [s for _, s in sp]},
    }


def make_split(n: int, seed: int, level: str = "hard", start: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    return pd.DataFrame([flatten_example(make_example(rng, start + i, level)) for i in range(n)])
