from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
EVAL_DIR = ROOT_DIR / "eval"

PDF_PATH = DATA_DIR / "employment-act-1955.pdf"
SECTIONS_PATH = DATA_DIR / "sections.json"
COLLECTION_NAME = "employment_act"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

GENERATOR_MODEL = "gpt-5-nano"
CHECKER_MODEL = "google/gemini-2.5-flash-lite"
JUDGE_MODEL = "anthropic/claude-haiku-4.5"

TOP_K = 5
MAX_REWRITES = 1
MAX_REGENS = 1
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"