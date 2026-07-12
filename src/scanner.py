"""Prompt assembly + grounding flow for Cellar Scanner. See
CELLAR_SCANNER_SPEC.md §2-3.
"""
from __future__ import annotations

from context import ContextBudgeter, Section
from guardrails import ConfidenceGate, ground_claims
from retriever import retrieve
from schemas import Recommendation, TastingNote, WineCard

SYSTEM_RULES = """You are Cellar Scanner, a wine recommendation assistant.
You will be given a wine card (extracted from a label photo) and similar
wines retrieved from a review database. Optionally a tasting note is given.

Rules you MUST follow:
- Base every claim in `profile` and every `Pairing.why` ONLY on the retrieved
  reviews given to you. Do not invent flavor notes, scores, or facts not
  present in the retrieved text.
- citations: list the ids of the retrieved reviews you actually drew from.
  Never invent a citation id.
- pairings: 2-3 food pairings, each grounded in the wine's actual profile
  (body, acidity, tannin, sweetness) as described in the retrieved reviews.
- similar_wines: titles of retrieved wines that are genuinely similar in
  style, not just the same variety.
"""

WINE_CONFIDENCE_GATE = ConfidenceGate(threshold=0.6)


def build_retrieval_query(card: WineCard, note: TastingNote | None) -> str:
    parts = [p for p in [card.wine_name, card.variety, card.region, card.producer] if p]
    if note:
        parts.append(note.text)
    return ", ".join(parts) or "wine"


def assemble_prompt(card: WineCard, note: TastingNote | None, *, total_budget: int = 2200):
    query = build_retrieval_query(card, note)
    retrieved = retrieve(query, k=8)
    retrieved_ids = [c["id"] for c in retrieved]

    budgeter = ContextBudgeter(total_budget=total_budget)
    budgeter.add(Section(name="system", items=[SYSTEM_RULES], priority=0, min_tokens=350, max_tokens=400))
    card_text = f"Wine card:\n{card.model_dump_json(indent=2)}"
    if note:
        card_text += f"\n\nTasting note:\n{note.text}"
    budgeter.add(Section(name="card", items=[card_text], priority=0, min_tokens=250, max_tokens=300))
    budgeter.add(Section(
        name="reviews",
        items=[f"[{c['id']}] {c['text']}" for c in retrieved],
        priority=2, max_tokens=1400,
    ))
    budgeter.add(Section(
        name="output_format",
        items=["Respond with ONLY JSON matching Recommendation: profile, pairings "
               "(list of {dish, why, citation_ids}), similar_wines, citations."],
        priority=1, min_tokens=100,
    ))

    packed = budgeter.pack()
    return packed.prompt, packed, retrieved_ids, [c["text"] for c in retrieved]


def ground_recommendation(rec: Recommendation, retrieved_texts: list[str]) -> Recommendation:
    """Post-hoc grounding filter: drop any pairing `why` that doesn't ground
    against the retrieved reviews. Returns a possibly-shortened Recommendation
    (never raises -- caller reports grounding_rate from this).
    """
    claims = [rec.profile] + [p.why for p in rec.pairings]
    report = ground_claims(claims, retrieved_texts)
    grounded_set = set(report.grounded)

    kept_pairings = [p for p in rec.pairings if p.why in grounded_set]
    profile = rec.profile if rec.profile in grounded_set else (
        rec.profile if not report.ungrounded else "(profile could not be fully grounded in retrieved reviews)"
    )
    return Recommendation(
        profile=profile, pairings=kept_pairings,
        similar_wines=rec.similar_wines, citations=rec.citations,
    ), report
