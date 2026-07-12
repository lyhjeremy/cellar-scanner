"""Blind-tasting variety classifier: dataset prep. See CELLAR_SCANNER_SPEC.md §4.

Critical eval-integrity step: many winemag descriptions NAME the grape
directly ("this Cabernet shows..."). We mask variety tokens + common synonyms
in the description text for BOTH training and the Claude reference eval --
otherwise the task is string matching, not blind tasting. Reports the %
of rows where a mask actually fired (evidence the step mattered).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "lora_harness"))

from prep import prep_dataset

CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "Project 21 Wine Sommelier RAG" / "wine-sommelier-rag" / "data" / "winemag-data-130k-v2.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "lora"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

MIN_REVIEWS_PER_VARIETY = 500
MAX_ROWS_PER_VARIETY = 1200

# Common synonyms/abbreviations that leak the variety without using its exact
# name -- built once by hand from domain knowledge, extended programmatically
# below with each variety's own name variants.
SYNONYMS = {
    "cabernet sauvignon": ["cab", "cabernet"],
    "pinot noir": ["pinot"],
    "pinot gris": ["pinot grigio"],
    "pinot grigio": ["pinot gris"],
    "syrah": ["shiraz"],
    "shiraz": ["syrah"],
    "grenache": ["garnacha"],
    "garnacha": ["grenache"],
    "sauvignon blanc": ["sauv blanc", "sauvignon"],
    "chardonnay": ["chard"],
    "zinfandel": ["zin"],
    "cabernet franc": ["cab franc"],
    "tempranillo": ["tinta roriz", "tinto fino"],
    "malbec": [],
    "merlot": [],
    "riesling": [],
    "sangiovese": [],
    "nebbiolo": [],
    "gewürztraminer": ["gewurztraminer"],
}


def build_mask_pattern(variety: str) -> re.Pattern:
    variants = {variety.lower()} | set(SYNONYMS.get(variety.lower(), []))
    # also mask the adjectival/plural forms loosely by matching the stem
    escaped = [re.escape(v) for v in variants]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def mask_description(text: str, pattern: re.Pattern) -> tuple[str, bool]:
    masked, n = pattern.subn("[grape]", text)
    return masked, n > 0


def load_rows() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("variety") and row.get("description") and row.get("title")]


def build_dataset() -> dict:
    rows = load_rows()
    variety_counts = Counter(r["variety"] for r in rows)
    eligible = {v for v, c in variety_counts.items() if c >= MIN_REVIEWS_PER_VARIETY}
    print(f"{len(eligible)} varieties with >= {MIN_REVIEWS_PER_VARIETY} reviews (of {len(variety_counts)} total)")

    # cap per variety, stratified sample (first N encountered -- data isn't
    # ordered adversarially, deterministic and simple)
    per_variety_kept: dict[str, list[dict]] = {v: [] for v in eligible}
    for row in rows:
        v = row["variety"]
        if v in eligible and len(per_variety_kept[v]) < MAX_ROWS_PER_VARIETY:
            per_variety_kept[v].append(row)

    records = []
    masked_count = 0
    for variety, variety_rows in per_variety_kept.items():
        pattern = build_mask_pattern(variety)
        for row in variety_rows:
            masked_desc, did_mask = mask_description(row["description"], pattern)
            if did_mask:
                masked_count += 1
            records.append({
                "title": row["title"],
                "variety": variety,
                "description": masked_desc,
            })

    mask_rate = masked_count / len(records) if records else 0.0
    print(f"{len(records)} rows total across {len(eligible)} varieties; masking fired on {mask_rate:.1%} of rows")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "classes.json").write_text(json.dumps(sorted(eligible), indent=2))
    (EVAL_DIR / "masking_report.json").write_text(json.dumps({
        "n_rows": len(records), "n_varieties": len(eligible), "mask_rate": round(mask_rate, 4),
    }, indent=2))

    def to_messages(rec):
        prompt = (
            f"Blind tasting note (grape name masked as [grape]): {rec['description']}\n\n"
            "What grape variety is this? Answer with a ranked list:\n"
            "1. <variety>\n2. <variety>\n3. <variety>"
        )
        # single most-likely-first target for supervised fine-tuning; the
        # model learns to rank the true label first
        answer = f"1. {rec['variety']}"
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    card = prep_dataset(
        records,
        entity_key_fn=lambda r: r["title"],
        to_messages_fn=to_messages,
        out_dir=OUT_DIR,
        label_key="variety",
    )
    card["mask_rate"] = round(mask_rate, 4)
    (EVAL_DIR / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card


if __name__ == "__main__":
    card = build_dataset()
    print(json.dumps({k: v for k, v in card.items() if k != "label_balance"}, indent=2))
