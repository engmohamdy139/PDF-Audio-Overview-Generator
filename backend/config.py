"""Centralized configuration for the PDF Audio Generator project.

Configuration values follow standard precedence:
1. Existing ambient / system environment variables (highest precedence).
2. Values defined in a local `.env` file at the project root (via python-dotenv).
3. In-code default values (fallback).
"""

import os
from pathlib import Path
from typing import Any

# Project root directory (taskv1)
BASE_DIR = Path(__file__).resolve().parents[1]

# Load local .env file if python-dotenv is installed and .env exists
# Note: load_dotenv() defaults to override=False, preserving ambient environment variables.
try:
    from dotenv import load_dotenv

    env_path = BASE_DIR / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def _get_int(key: str, default: int) -> int:
    """Safely parse an integer environment variable with fallback."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# --- Ollama / LLM Configuration ---
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:3b-instruct")
NUM_CTX: int = _get_int("NUM_CTX", 4096)

# --- Document Chunking Configuration ---
CHUNK_TARGET_SIZE: int = _get_int("CHUNK_TARGET_SIZE", 5000)
CHUNK_MAX_SIZE: int = _get_int("CHUNK_MAX_SIZE", 6000)

# --- Text-to-Speech (gTTS) Configuration ---
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "en")

# --- OCR Configuration (Optional) ---
INDRA_OCR_MODEL_DIR: str | None = os.getenv("INDRA_OCR_MODEL_DIR") or None


def get_config_dict() -> dict[str, Any]:
    """Return a dictionary view of current configuration values."""
    return {
        "OLLAMA_URL": OLLAMA_URL,
        "MODEL_NAME": MODEL_NAME,
        "NUM_CTX": NUM_CTX,
        "CHUNK_TARGET_SIZE": CHUNK_TARGET_SIZE,
        "CHUNK_MAX_SIZE": CHUNK_MAX_SIZE,
        "TTS_LANGUAGE": TTS_LANGUAGE,
        "INDRA_OCR_MODEL_DIR": INDRA_OCR_MODEL_DIR,
    }
