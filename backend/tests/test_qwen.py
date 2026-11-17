
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import requests

from backend.config import MODEL_NAME, NUM_CTX, OLLAMA_URL


def summarize_text(text: str) -> str:
    prompt = f"""
You are an assistant that explains research papers simply.

Summarize the following text in clear English.
Preserve the main ideas and do not invent information.

TEXT:
{text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": NUM_CTX,
            },
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    sample_text = """
    The Transformer is a neural network architecture based on attention.
    It does not rely on recurrent or convolutional layers.
    """

    print(summarize_text(sample_text))
