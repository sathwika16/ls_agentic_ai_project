import json
import pickle
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import INDEX_FILE, DATA_DIR

VEC_PATH = Path(DATA_DIR) / "vectorizer.pkl"
DOCS_CACHE = Path(DATA_DIR) / "docs_cache.pkl"

def _load_records() -> List[dict]:
    path = Path(INDEX_FILE)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records

def _ensure_vectorizer(records: List[dict]) -> Tuple[TfidfVectorizer, List[str]]:
    texts = [r.get("text", "") for r in records]
    if not texts:
        return None, []
    try:
        if VEC_PATH.exists() and DOCS_CACHE.exists():
            with open(VEC_PATH, "rb") as f:
                vectorizer = pickle.load(f)
            with open(DOCS_CACHE, "rb") as f:
                docs_texts = pickle.load(f)
            if len(docs_texts) == len(texts):
                return vectorizer, docs_texts
    except Exception:
        pass
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    docs_texts = texts
    vectorizer.fit(docs_texts)
    VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(DOCS_CACHE, "wb") as f:
        pickle.dump(docs_texts, f)
    return vectorizer, docs_texts

def retrieve_docs(query: str, k: int = 4) -> str:
    query = (query or "").strip()
    if not query:
        return ""
    records = _load_records()
    if not records:
        return ""
    vectorizer, docs_texts = _ensure_vectorizer(records)
    if vectorizer is None:
        return ""
    q_vec = vectorizer.transform([query])
    docs_mat = vectorizer.transform(docs_texts)
    sims = cosine_similarity(q_vec, docs_mat).flatten()
    scored = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
    parts = []
    for idx, score in scored:
        if score <= 0:
            continue
        rec = records[idx]
        parts.append(f"Score: {score:.3f} | Source: {rec.get('source','unknown')} | Page: {rec.get('page','n/a')}\n{rec.get('text','')}")
    return "\n\n---\n\n".join(parts)