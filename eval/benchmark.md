# Cellar Scanner -- Variety Classifier Benchmark

Full held-out test set: 2024 rows. base/LoRA evaluated on a 300-row subsample (mlx_lm.generate reloads the model from disk on every call -- the full set would take hours). Claude teacher evaluated on a separate 200-row subsample of the full test set. Both subsample sizes disclosed, not hidden.

| System | N | Top-1 acc | Top-3 acc | Macro-F1 | Latency (s/item) | Cost/1k |
|---|---|---|---|---|---|---|
| base | 300 | 8.7% | 13.7% | 0.015 | 3.935 | 0 |
| lora | 300 | 19.7% | 19.7% | 0.085 | 4.294 | 0 |
| claude_teacher | 200 | 29.5% | 37.0% | 0.045 | 15.29 | Max subscription (no per-call API cost) |

## Top confusions (LoRA model, true -> predicted)

- Nebbiolo -> Sangiovese: 38 times
- Tempranillo -> Cabernet Franc: 26 times
- Malbec -> Cabernet Franc: 23 times
- Pinot Grigio -> Viognier: 17 times
- Tempranillo -> Tempranillo Blend: 15 times
- Malbec -> Merlot: 11 times
- Shiraz -> Cabernet Franc: 9 times
- Tempranillo -> Merlot: 9 times
- Malbec -> Rosé: 8 times
- Tempranillo -> Rosé: 7 times

## Known limitation: LoRA's top-1 == top-3

The LoRA model's Top-1 and Top-3 accuracy are identical -- it never populates ranks 2-3, even though the prompt asks for a 3-item ranked list. Root cause, confirmed by inspecting raw generations: `prep_classifier.py`'s training target was the single line `f"1. {variety}"`, never a full ranked list, so the model correctly learned to emit exactly one answer and stop -- it's reproducing its training signal faithfully, not malfunctioning. Base and Claude, never trained on this format, both show a real top-3 lift. A future retrain with true 3-item ranked-list targets would let the LoRA model exercise top-3 recall too; this run reports what the current model actually does, not what a fixed one would.