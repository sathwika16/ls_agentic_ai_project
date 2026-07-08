# Student Research Copilot — Week 4 (Multi-Agent Capstone)

Lightweight multi-agent system for ingesting PDFs, performing semantic retrieval, generating revision plans, answering questions, validating grounding, and handling fallbacks/human review.

## Features
- Supervisor orchestration (LangGraph)
- Planner, Retriever, Tutor, Validator, Fallback, Human-review agents
- PDF ingestion -> JSONL index
- TF-IDF semantic retrieval (scikit-learn) with cosine similarity
- Human-editable checkpoint for low-confidence responses
- Failure handling: retries, fallback, human checkpoint

## Repo structure
- app.py — Streamlit UI
- graph.py — LangGraph orchestration
- agents.py — LLM prompts + wrappers (Gemini integration)
- ingest.py — PDF -> JSONL ingestion
- tools.py — retrieval (vectorizer + similarity)
- config.py — env loader
- requirements.txt — deps
- .env.example — example env vars
- data/ — generated index and vectorizer cache
- chroma_db/ uploads/ — optional storage paths

## Requirements
- Windows, Python 3.11+
- Virtualenv recommended

## Environment (.env)
Create `week4/.env` (do NOT commit). Example:
```
GEMINI_API_KEY=your_gemini_key_here
CHROMA_DIR=./chroma_db
UPLOAD_DIR=./uploads
DATA_DIR=./data
INDEX_FILE=./data/index.jsonl
```
Important: rotate/revoke any exposed keys immediately.

## Install & run
Open PowerShell:
```
cd "C:\Users\saray\OneDrive\Desktop\ls_agentic_ai\week4"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If you switched to Gemini, ensure `langchain-google-genai` is installed and the chosen interpreter in VS Code is `week4\.venv\Scripts\python.exe`.

## Usage
1. Upload a PDF in the sidebar → click "Ingest PDF" (this overwrites index by default).
2. Enter a query or request a revision plan in the main panel.
3. Click "Run".
4. Review the answer, edit in the human-review box if required.

## Testing tips
- Test with short queries that match content in your PDF.
- Try: "Make a 5-day revision plan for [topic]" and "Explain [concept] in simple terms".

## Notes & recommendations
- Current retrieval uses TF-IDF for simplicity. For production-quality RAG, replace with embeddings + Chroma (use Google or HuggingFace embeddings).
- Validation threshold is intentionally conservative; adjust VALIDATION_THRESHOLD in graph.py as needed.
- Ingest overwrites index by default to avoid duplicates; set overwrite=False in ingest_pdf if you prefer append mode.
- Log exceptions in `_safe` helpers for easier debugging.

## Troubleshooting
- Pylance import errors: ensure VS Code uses the project venv interpreter.
- Missing key errors: verify `.env` is in `week4/` and contains GEMINI_API_KEY.
- Quota errors: rotate keys or switch to a local model.

## License
MIT — adapt for your capstone submission.
