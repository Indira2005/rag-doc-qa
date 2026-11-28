import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Optional backup for the app. Off by default; set LLM_FALLBACK=gemini to turn it on.
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "").strip().lower()
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
REDIS_URL = os.getenv("REDIS_URL", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATES = int(os.getenv("CANDIDATES", "20"))
# Optional score cutoff. 0 = off: the LLM decides whether the passages answer the question.
# It is off because valid and unrelated questions scored too close together in testing.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0"))
