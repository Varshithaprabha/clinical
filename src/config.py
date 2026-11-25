"""Central paths and settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIMA_CSV = ROOT / "diabetes.csv"
DIABD_CSV = ROOT / "DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "models" / "cache"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

BIOBERT_NAME = os.getenv("BIOBERT_MODEL", "dmis-lab/biobert-base-cased-v1.1")
BIOBERT_PCA_COMPONENTS = 16

# LLM (any OpenAI-style provider works; Groq is used by default because a key is configured)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

MGDL_PER_MMOL = 18.016

for _d in (MODELS_DIR, REPORTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
