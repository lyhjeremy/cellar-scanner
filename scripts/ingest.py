"""Build the 30k stratified-by-variety wine review index. Adapted from repo
21's ingest.py recipe (CELLAR_SCANNER_SPEC.md §3), but stratified by variety
rather than sorted by rating -- gives even retrieval coverage across wine
types instead of skewing to top-rated reviews only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "Project 21 Wine Sommelier RAG" / "wine-sommelier-rag" / "data" / "winemag-data-130k-v2.csv"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION = "wine_reviews"
TARGET_SIZE = 30_000


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["description", "title", "variety"]).copy()
    df = df.drop_duplicates(subset=["title", "description"])
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """Water-filling stratified sample: with 707 varieties (many with a
    handful of reviews each), a flat target_size/n_varieties quota starves
    on the long tail and undershoots target_size badly (10k vs a 30k goal,
    caught by inspecting ingest.py's own output). Redistribute unused quota
    from small varieties to larger ones until target_size is reached or the
    whole corpus is exhausted.
    """
    counts = df.groupby("variety").size().sort_values()
    remaining_budget = target_size
    remaining_varieties = len(counts)
    quota = {}
    for variety, count in counts.items():
        share = max(1, remaining_budget // remaining_varieties)
        take = min(count, share)
        quota[variety] = take
        remaining_budget -= take
        remaining_varieties -= 1

    parts = [
        g.sample(min(len(g), quota[variety]), random_state=42)
        for variety, g in df.groupby("variety", group_keys=False)
    ]
    return pd.concat(parts).reset_index(drop=True)


def _region(row: pd.Series) -> str:
    parts = [row.get("region_1"), row.get("province"), row.get("country")]
    parts = [str(p) for p in parts if isinstance(p, str) and p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)


def _document(row: pd.Series) -> str:
    origin = _region(row)
    header = f"{row['title']} — {row['variety']}"
    if origin:
        header += f" from {origin}"
    return f"{header}.\n{row['description']}"


def _metadata(row: pd.Series) -> dict:
    def s(v):
        return v if isinstance(v, str) and v.strip() else ""

    meta = {
        "title": s(row.get("title")), "variety": s(row.get("variety")),
        "winery": s(row.get("winery")), "country": s(row.get("country")),
        "province": s(row.get("province")), "region": s(row.get("region_1")),
    }
    pts, prc = row.get("points"), row.get("price")
    if isinstance(pts, (int, float)) and not math.isnan(pts):
        meta["points"] = int(pts)
    if isinstance(prc, (int, float)) and not math.isnan(prc):
        meta["price"] = float(prc)
    return meta


def build() -> int:
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"Loading {CSV_PATH.name}...")
    df = _clean(pd.read_csv(CSV_PATH))
    df = _stratified_sample(df, TARGET_SIZE)
    print(f"Indexing {len(df):,} reviews, stratified across {df['variety'].nunique()} varieties...")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer("all-MiniLM-L6-v2")
    batch = 512
    total = len(df)
    for start in range(0, total, batch):
        chunk = df.iloc[start:start + batch]
        docs = [_document(r) for _, r in chunk.iterrows()]
        metas = [_metadata(r) for _, r in chunk.iterrows()]
        ids = [f"wine-{start + i}" for i in range(len(chunk))]
        embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        print(f"  {min(start + batch, total):,}/{total:,}")

    print(f"Done. Collection '{COLLECTION}' holds {coll.count():,} reviews.")
    return coll.count()


if __name__ == "__main__":
    build()
