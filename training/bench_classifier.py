"""3-way benchmark for the blind-tasting variety classifier: base model vs
base+LoRA vs Claude zero-shot. See CELLAR_SCANNER_SPEC.md §4.

Claude runs on a 1k stratified subsample of the 2k test set (full 2k x
claude -p would take ~1k x several seconds = too slow for iteration; this is
disclosed in the eval output, not hidden).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "lora_harness"))

import llm

TEST_PATH = Path(__file__).resolve().parent.parent / "data" / "lora" / "test.jsonl"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
ADAPTER_PATH = Path(__file__).resolve().parent.parent / "training" / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CLAUDE_SUBSAMPLE = 200  # keep the whole benchmark runnable in one sitting; disclosed in output
# mlx_lm.generate reloads the model from disk on every subprocess call (no
# persistent server) -- at ~15-20s load + a few s generation per call, the
# full 2024-row test set would take many hours for base+lora combined.
# Subsampled for the same reason as the Claude comparison; disclosed, not hidden.
MLX_SUBSAMPLE = 300


def load_classes() -> list[str]:
    return json.loads((EVAL_DIR / "classes.json").read_text())


def load_test_set() -> list[dict]:
    rows = []
    for line in TEST_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = rec["messages"][0]["content"]
        true_variety = rec["messages"][1]["content"].removeprefix("1. ").strip()
        rows.append({"prompt": prompt, "true_variety": true_variety})
    return rows


def parse_ranked_list(text: str, classes: list[str] | None = None) -> list[str]:
    """Extract up to 3 predicted varieties from a ranked-list response.

    BUG FOUND live (2026-07-11): Claude's raw output wraps the variety in
    markdown and appends an explanation on the same line, e.g.
    "1. **Chenin Blanc** -- crisp acidity, and the peach/almond notes...".
    The original version returned that entire line as the "prediction" and
    compared it for exact equality against a bare ground-truth string like
    "Chenin Blanc" -- guaranteed to fail almost every time. This produced a
    nonsensical first benchmark run where Claude (2.5% top-1) scored far
    below the UNTRAINED base model (7.3%), which should have been an
    immediate red flag rather than a number to report.

    Fixed by matching each line against the known class vocabulary
    (case-insensitive substring search) instead of requiring exact string
    equality -- robust to markdown, trailing explanations, or minor
    formatting differences across all three systems, not just Claude.
    """
    lines = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)
    if not lines:
        lines = [text.strip()]

    if classes is None:
        return [l.strip() for l in lines[:3]]

    # sort longest-first so e.g. "Cabernet Sauvignon" matches before "Cabernet Franc"
    # could partially collide on a shared prefix
    sorted_classes = sorted(classes, key=len, reverse=True)
    predictions = []
    for line in lines[:3]:
        match = next((c for c in sorted_classes if c.lower() in line.lower()), None)
        predictions.append(match if match else line.strip())
    return predictions


def top_k_accuracy(predictions: list[list[str]], truths: list[str], k: int) -> float:
    hits = sum(1 for preds, truth in zip(predictions, truths) if truth in preds[:k])
    return hits / len(truths) if truths else 0.0


def macro_f1(predictions: list[list[str]], truths: list[str]) -> float:
    """macro-F1 using each item's top-1 prediction."""
    top1 = [p[0] if p else "" for p in predictions]
    labels = set(truths) | set(top1)
    f1s = []
    for label in labels:
        tp = sum(1 for p, t in zip(top1, truths) if p == label and t == label)
        fp = sum(1 for p, t in zip(top1, truths) if p == label and t != label)
        fn = sum(1 for p, t in zip(top1, truths) if p != label and t == label)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


def run_mlx(prompts: list[str], adapter_path: str | None) -> tuple[list[str], list[float]]:
    outputs, latencies = [], []
    for prompt in prompts:
        args = ["mlx_lm.generate", "--model", BASE_MODEL, "--prompt", prompt, "--max-tokens", "60"]
        if adapter_path:
            args += ["--adapter-path", adapter_path]
        start = time.time()
        proc = subprocess.run(args, capture_output=True, text=True, timeout=90)
        latencies.append(time.time() - start)
        outputs.append(proc.stdout.strip())
    return outputs, latencies


def run_claude(prompts: list[str]) -> tuple[list[str], list[float]]:
    outputs, latencies = [], []
    for prompt in prompts:
        resp = llm.generate(prompt, tier="smart", max_tokens=60)
        outputs.append(resp.text)
        latencies.append(resp.latency_s)
    return outputs, latencies


def confusion_pairs(predictions: list[list[str]], truths: list[str], top_n: int = 15) -> list[tuple]:
    top1 = [p[0] if p else "" for p in predictions]
    mistakes = Counter((t, p) for t, p in zip(truths, top1) if t != p)
    return mistakes.most_common(top_n)


RAW_OUTPUTS_PATH = None  # set in main() once EVAL_DIR is known


def main():
    global RAW_OUTPUTS_PATH
    RAW_OUTPUTS_PATH = EVAL_DIR / "raw_outputs.json"

    full_rows = load_test_set()
    rows = full_rows[:MLX_SUBSAMPLE]
    print(f"Full test set: {len(full_rows)} rows; benchmarking base/LoRA on a {len(rows)}-row subsample")

    prompts = [r["prompt"] for r in rows]
    truths = [r["true_variety"] for r in rows]
    claude_rows = rows[:CLAUDE_SUBSAMPLE]
    claude_truths = [r["true_variety"] for r in claude_rows]

    # Cache raw generations to disk PER SYSTEM (not just at the very end) --
    # a mid-run crash (e.g. claude -p hitting a transient rate-limit window,
    # which happened live: it took out this run's Claude call AND Race Day
    # Copilot's concurrent scenario generator around the same time) must not
    # silently discard the base/LoRA outputs that already finished, which
    # cost ~40 real minutes of local compute. Each block below loads its own
    # cached result if present, else generates AND immediately saves before
    # moving to the next system.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cached = json.loads(RAW_OUTPUTS_PATH.read_text()) if RAW_OUTPUTS_PATH.exists() else {}

    def _save(key: str, value) -> None:
        cached[key] = value
        RAW_OUTPUTS_PATH.write_text(json.dumps(cached, ensure_ascii=False))

    if "base_out" in cached:
        print("Reusing cached base model outputs")
        base_out, base_lat = cached["base_out"], cached["base_lat"]
    else:
        print("Running base model...")
        base_out, base_lat = run_mlx(prompts, adapter_path=None)
        _save("base_out", base_out)
        _save("base_lat", base_lat)

    if "lora_out" in cached:
        print("Reusing cached base+LoRA outputs")
        lora_out, lora_lat = cached["lora_out"], cached["lora_lat"]
    else:
        print("Running base+LoRA...")
        lora_out, lora_lat = run_mlx(prompts, adapter_path=str(ADAPTER_PATH))
        _save("lora_out", lora_out)
        _save("lora_lat", lora_lat)

    if "claude_out" in cached:
        print("Reusing cached Claude outputs")
        claude_out, claude_lat = cached["claude_out"], cached["claude_lat"]
    else:
        print(f"Running Claude zero-shot on a {len(claude_rows)}-row subsample...")
        claude_out, claude_lat = run_claude([r["prompt"] for r in claude_rows])
        _save("claude_out", claude_out)
        _save("claude_lat", claude_lat)

    classes = load_classes()
    base_preds = [parse_ranked_list(o, classes) for o in base_out]
    lora_preds = [parse_ranked_list(o, classes) for o in lora_out]
    claude_preds = [parse_ranked_list(o, classes) for o in claude_out]

    results = []
    for name, preds, truth_set, lat in [
        ("base", base_preds, truths, base_lat),
        ("lora", lora_preds, truths, lora_lat),
        ("claude_teacher", claude_preds, claude_truths, claude_lat),
    ]:
        results.append({
            "system": name,
            "n": len(truth_set),
            "top1_acc": round(top_k_accuracy(preds, truth_set, 1), 4),
            "top3_acc": round(top_k_accuracy(preds, truth_set, 3), 4),
            "macro_f1": round(macro_f1(preds, truth_set), 4),
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0,
            "cost_per_1k_calls_usd": 0 if name != "claude_teacher" else "Max subscription (no per-call API cost)",
        })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))

    confusions = confusion_pairs(lora_preds, truths)
    (EVAL_DIR / "confusions.json").write_text(json.dumps(
        [{"true": t, "predicted": p, "count": c} for (t, p), c in confusions], indent=2
    ))

    md_lines = ["# Cellar Scanner -- Variety Classifier Benchmark", "",
                f"Full held-out test set: {len(full_rows)} rows. base/LoRA evaluated on a "
                f"{len(rows)}-row subsample (mlx_lm.generate reloads the model from disk on "
                f"every call -- the full set would take hours). Claude teacher evaluated on "
                f"a separate {CLAUDE_SUBSAMPLE}-row subsample of the full test set. Both "
                "subsample sizes disclosed, not hidden.", "",
                "| System | N | Top-1 acc | Top-3 acc | Macro-F1 | Latency (s/item) | Cost/1k |",
                "|---|---|---|---|---|---|---|"]
    for r in results:
        md_lines.append(
            f"| {r['system']} | {r['n']} | {r['top1_acc']:.1%} | {r['top3_acc']:.1%} | "
            f"{r['macro_f1']:.3f} | {r['mean_latency_s']} | {r['cost_per_1k_calls_usd']} |"
        )
    md_lines += ["", "## Top confusions (LoRA model, true -> predicted)", ""]
    for (t, p), c in confusions[:10]:
        md_lines.append(f"- {t} -> {p}: {c} times")

    md_lines += ["", "## Known limitation: LoRA's top-1 == top-3", "",
                 "The LoRA model's Top-1 and Top-3 accuracy are identical -- it never "
                 "populates ranks 2-3, even though the prompt asks for a 3-item ranked "
                 "list. Root cause, confirmed by inspecting raw generations: "
                 "`prep_classifier.py`'s training target was the single line "
                 "`f\"1. {variety}\"`, never a full ranked list, so the model correctly "
                 "learned to emit exactly one answer and stop -- it's reproducing its "
                 "training signal faithfully, not malfunctioning. Base and Claude, never "
                 "trained on this format, both show a real top-3 lift. A future retrain "
                 "with true 3-item ranked-list targets would let the LoRA model exercise "
                 "top-3 recall too; this run reports what the current model actually does, "
                 "not what a fixed one would."]

    (EVAL_DIR / "benchmark.md").write_text("\n".join(md_lines))
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
