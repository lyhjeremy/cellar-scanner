"""Query the 30k stratified wine-review chroma collection."""
from __future__ import annotations

from pathlib import Path

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION = "wine_reviews"

_collection = None
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION)
    return _collection


def retrieve(query: str, k: int = 8) -> list[dict]:
    """Returns [{id, text, metadata}] ranked by relevance to `query`."""
    coll = _get_collection()
    query_emb = _get_model().encode(query, normalize_embeddings=True).tolist()
    result = coll.query(query_embeddings=[query_emb], n_results=k)
    cards = []
    for i in range(len(result["ids"][0])):
        cards.append({
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
        })
    return cards
