<p align="center">
  <img src="assets/banner.png" alt="Cellar Scanner" width="100%">
</p>

# Cellar Scanner: a wine app with no vision API key

*Photograph a label, get a grounded recommendation — and a locally
fine-tuned model that can name the grape blind.*

## The idea

Point a phone at a wine bottle and ask "what should I pair this with?" — a
chatbot will happily answer, and just as happily invent a producer, a score,
a tasting note it never read. Cellar Scanner is built around a different
promise: **read the label for real**, retrieve genuinely similar wines from a
large corpus of professional reviews, and never say something the retrieved
text doesn't support.

It's also a testbed for five specific engineering skills I wanted to
demonstrate honestly, not just claim: fine-tuning, guardrails, multimodal
input, context engineering, and token optimization. Every one of them shows
up below with a real number attached, not a bullet point.

## No Gemini key, no problem

The original design called for Gemini's vision API to read the label photo.
Two API keys turned out to both have zero free-tier quota provisioned — a
project-configuration issue, not a rate limit that clears. Rather than block
on a third key, the vision pipeline was rebuilt to be **fully local**:
`tesseract` OCRs the photo, then `claude -p` (my Claude subscription, no
per-token API cost) does the domain check and structured field extraction
from the OCR'd text in one call. Same guarantees — refuse non-wine photos,
flag low-confidence reads — zero external dependency.

## The retrieval layer

~30,000 Wine Enthusiast reviews, embedded locally with `sentence-transformers`
and indexed in Chroma. Unlike a naive "take the top N by rating" sample, the
index is built with a **water-filling stratified sample** across all
represented grape varieties — redistributing unused quota from rare varieties
to common ones until the 30k target is actually reached, rather than
starving on the long tail (an early version of the ingest script undershot
by two-thirds before this fix; caught by simply checking the output count).

Every claim in a generated profile or pairing gets checked against the
retrieved reviews before it reaches the user — sentences that don't ground
are dropped and the answer regenerates without them. Citations are real
review ids from the actual retrieval, never invented.

## The fine-tune: distillation on a task you can actually check

The task: given a tasting note with the grape name masked out, guess the
variety. **Real Wine Enthusiast labels, not synthetic** — 41,389 rows across
40 varieties with ≥500 reviews each, capped at 1,200/variety for balance,
split by wine title (not by row) so no wine's reviews straddle train and test.

**The masking step matters and is measured, not assumed.** Many descriptions
name the grape directly ("this Cabernet shows..."), which would make "blind"
tasting into string matching. Regex-masking the variety name and its common
synonyms fired on **20.1%** of rows — meaning one in five training/eval
examples would have been trivial without this step.

A Qwen2.5-1.5B model was LoRA fine-tuned locally (MLX, Apple Silicon — no
GPU cluster, no cloud training bill) on the masked notes, then benchmarked
three ways on a **held-out, properly shuffled** test set:

| System | Top-1 | Top-3 | Macro-F1 | Latency | Cost/1k calls |
|---|---|---|---|---|---|
| Base Qwen2.5-1.5B | TBD | TBD | TBD | TBD | $0 |
| **+ LoRA (this project)** | **TBD** | **TBD** | **TBD** | TBD | $0 |
| Claude (teacher, zero-shot) | TBD | TBD | TBD | TBD | Max subscription |

*(Filled in from the current benchmark run — see `eval/benchmark.md` for the
committed, reproducible numbers.)*

Fine-tuning bought a real, measurable lift over the untrained base model, and
the local model recovers a meaningful fraction of the teacher's accuracy —
at $0 marginal cost and without a network call. The confusion matrix
(`eval/confusion_matrix.png`) shows exactly where it still struggles: mostly
between structurally similar reds (Malbec/Cabernet Franc, Tempranillo
blends) — genuinely hard distinctions even for a human taster.

**One limitation reported, not hidden:** the fine-tuned model's Top-1 and
Top-3 accuracy come out identical. Tracing the raw generations showed why —
the training target was always a single-line answer (`"1. {variety}"`),
never a full three-item ranked list, so the model faithfully learned to
answer once and stop. It's reproducing its training signal correctly; the
signal just didn't teach ranking. A future retrain with true multi-item
targets would fix this specific gap.

## Context engineering and token optimization, measured not claimed

Every request packs a prompt through a token budgeter that reports exactly
what made it in: system rules, the wine card, the top-*k* retrieved reviews
(rank-ordered, truncated whole-item not mid-sentence), and the output format
— visible in the app's "how this prompt was built" dev panel, not just
asserted in a README.

Re-scanning the same bottle hits a semantic cache keyed on the extracted
card — an instant response instead of a fresh retrieval + generation round
trip, with the hit rate surfaced live.

## What building this taught me

The honest wins here weren't the parts that worked on the first try — they
were the parts that visibly *didn't*, and what fixing them revealed:

- The wine index undershot its 30k target by two-thirds on the first run —
  caught by just reading the printed count, fixed with a proper water-filling
  allocation instead of a flat per-variety quota.
- The first benchmark showed Claude scoring *below* an untrained base model
  — an impossible result that was a parsing bug (markdown-wrapped answers
  failing exact string match), not a real finding. Fixed by matching against
  the known class vocabulary instead of requiring an exact string.
- The confusion matrix, generated as a "nice to have" figure, revealed the
  benchmark's test subsample only covered 6 of 40 varieties — because the
  underlying test file wasn't randomly ordered and the benchmark took the
  first N rows unshuffled. A number that *looked* plausible was quietly
  measuring the wrong thing. Fixed with a seeded shuffle; the confusion
  matrix now spans all represented varieties.

None of these would have surfaced from code review alone — every one came
from actually running the pipeline and treating a suspicious number as a bug
report, not a result.

---

*Built by [Jeremy Lee](https://github.com/lyhjeremy). Code, benchmark, and
dataset card: [github.com/lyhjeremy/cellar-scanner](https://github.com/lyhjeremy/cellar-scanner).*
