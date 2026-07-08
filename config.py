from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db").strip().strip('"').strip("'")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads").strip().strip('"').strip("'")
DATA_DIR = os.getenv("DATA_DIR", "./data").strip().strip('"').strip("'")
INDEX_FILE = os.getenv("INDEX_FILE", "./data/index.jsonl").strip().strip('"').strip("'")