"""Re-score the cached benchmark outputs on the rows all three systems share.

bench_classifier.py runs the base and LoRA models on the first 300 rows of the
seeded test shuffle and Claude on the first 200 of those same rows
(`claude_rows = rows[:CLAUDE_SUBSAMPLE]`). So the 200 Claude rows are a subset
of the 300, and the LoRA and Claude answers can be compared note by note.

Nothing is re-generated: this reads eval/raw_outputs.json and the test split
that prep_classifier.py writes to data/lora/test.jsonl (git-ignored), and uses
bench_classifier's own loader and parser. Writes:

  eval/paired_lora_claude.json   top-1 on the shared 200 rows, the agreement
                                 counts and an exact sign test on the notes
                                 where only one system was right
  eval/blind_samples.json        the 200 shared notes (grape masked), the true
                                 variety and each system's top answer, for the
                                 blind-tasting quiz on the write-up page

Run: python training/paired_check.py
"""
from __future__ import annotations

import json
import re
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_classifier import CLAUDE_SUBSAMPLE, EVAL_DIR, load_classes, load_test_set, parse_ranked_list  # noqa: E402

NOTE_RE = re.compile(r"\):\s*(.*?)\n\nWhat grape variety", re.S)


def sign_test(a: int, b: int) -> float:
    """Two-sided exact binomial test of a vs b successes at p = 0.5."""
    n, k = a + b, max(a, b)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    raw = json.loads((EVAL_DIR / "raw_outputs.json").read_text())
    classes = load_classes()
    rows = load_test_set()[:CLAUDE_SUBSAMPLE]
    truths = [r["true_variety"] for r in rows]
    top1 = {k: [(parse_ranked_list(o, classes) or [""])[0] for o in raw[f"{k}_out"][:CLAUDE_SUBSAMPLE]]
            for k in ("base", "lora", "claude")}
    hit = {k: [p == t for p, t in zip(v, truths)] for k, v in top1.items()}
    both = sum(a and b for a, b in zip(hit["lora"], hit["claude"]))
    lora_only = sum(a and not b for a, b in zip(hit["lora"], hit["claude"]))
    claude_only = sum(b and not a for a, b in zip(hit["lora"], hit["claude"]))
    out = {
        "n_shared_rows": len(rows),
        "top1_on_shared_rows": {k: round(sum(v) / len(v), 4) for k, v in hit.items()},
        "both_right": both, "lora_only_right": lora_only, "claude_only_right": claude_only,
        "neither_right": len(rows) - both - lora_only - claude_only,
        "sign_test_p_two_sided": round(sign_test(lora_only, claude_only), 4),
    }
    (EVAL_DIR / "paired_lora_claude.json").write_text(json.dumps(out, indent=2))
    samples = []
    for r, t, b, l, c in zip(rows, truths, top1["base"], top1["lora"], top1["claude"]):
        m = NOTE_RE.search(r["prompt"])
        samples.append({"note": m.group(1).strip() if m else r["prompt"], "true": t,
                        "base": b, "lora": l, "claude": c})
    (EVAL_DIR / "blind_samples.json").write_text(json.dumps(samples, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
