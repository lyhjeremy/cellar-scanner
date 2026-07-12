"""Pydantic contracts for Cellar Scanner. See CELLAR_SCANNER_SPEC.md §1."""
from __future__ import annotations

from pydantic import BaseModel


class WineCard(BaseModel):
    producer: str | None = None
    wine_name: str | None = None
    vintage: int | None = None
    variety: str | None = None
    region: str | None = None
    country: str | None = None
    abv: float | None = None
    extraction_confidence: float


class TastingNote(BaseModel):
    text: str
    descriptors: list[str] = []


class Pairing(BaseModel):
    dish: str
    why: str
    citation_ids: list[str] = []


class Recommendation(BaseModel):
    profile: str
    pairings: list[Pairing]
    similar_wines: list[str] = []
    citations: list[str] = []
