import json
from pathlib import Path
from pypdf import PdfReader
from config import DATA_DIR, INDEX_FILE

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def ingest_pdf(pdf_path: str, overwrite: bool = True) -> int:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf_path)
    records = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for idx, chunk in enumerate(chunk_text(text)):
            records.append({
                "source": Path(pdf_path).name,
                "page": i + 1,
                "chunk": idx + 1,
                "text": chunk,
            })
    if not records:
        return 0
    mode = "w" if overwrite else "a"
    with open(INDEX_FILE, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)