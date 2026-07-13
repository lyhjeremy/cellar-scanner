"""Confusion matrix heatmap for the LoRA classifier, built from the cached
benchmark raw outputs -- no new LLM/local-generation calls needed."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_classifier import load_classes, load_test_set, parse_ranked_list, MLX_SUBSAMPLE

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def main():
    raw = json.loads((EVAL_DIR / "raw_outputs.json").read_text())
    rows = load_test_set()[:MLX_SUBSAMPLE]
    truths = [r["true_variety"] for r in rows]
    classes = load_classes()
    preds = [parse_ranked_list(o, classes)[0] if parse_ranked_list(o, classes) else "?" for o in raw["lora_out"]]

    # top-N varieties by test-set frequency, to keep the matrix readable
    top_varieties = [v for v, _ in Counter(truths).most_common(12)]
    idx = {v: i for i, v in enumerate(top_varieties)}
    matrix = np.zeros((len(top_varieties), len(top_varieties)))
    for t, p in zip(truths, preds):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(top_varieties))); ax.set_xticklabels(top_varieties, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_varieties))); ax.set_yticklabels(top_varieties, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Cellar Scanner LoRA -- confusion matrix (top 12 varieties by test frequency)")
    for i in range(len(top_varieties)):
        for j in range(len(top_varieties)):
            if matrix[i][j] > 0:
                ax.text(j, i, int(matrix[i][j]), ha="center", va="center",
                        fontsize=7, color="white" if matrix[i][j] > matrix.max() / 2 else "black")
    fig.colorbar(im, label="count")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved {EVAL_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
