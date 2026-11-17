
import json
from pathlib import Path
import sys

import requests


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import MODEL_NAME, NUM_CTX, OLLAMA_URL

PAGES_DIR = BASE_DIR / "outputs" / "large_pdf" / "pages"
OUTPUT_DIR = BASE_DIR / "outputs" / "large_pdf" / "summary"

# Conservative limits for a local 3B model
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500


def summarize_text(text: str, instruction: str) -> str:
    prompt = f"""
You are a careful research-paper assistant.

Task:
{instruction}

Rules:
- Use only information supported by the provided text.
- Preserve important technical terms, methods, findings, and limitations.
- Do not invent facts, results, citations, or conclusions.
- If information is unclear, say so.
- Write in clear English.

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
        timeout=600,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def load_pages() -> str:
    files = sorted(PAGES_DIR.glob("page_*.md"))

    if not files:
        raise FileNotFoundError(
            f"No page Markdown files found in: {PAGES_DIR}"
        )

    # Preserve the original page order.
    return "\n\n".join(
        f"[Page {i + 1}]\n{file.read_text(encoding='utf-8')}"
        for i, file in enumerate(files)
    )


def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks_dir = OUTPUT_DIR / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    state_path = OUTPUT_DIR / "state.json"
    final_path = OUTPUT_DIR / "final_summary.md"

    text = load_pages()
    chunks = split_into_chunks(text)

    state = {
        "total_chunks": len(chunks),
        "completed_chunks": [],
    }

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    completed = set(state.get("completed_chunks", []))

    print(f"Total characters: {len(text)}")
    print(f"Total chunks: {len(chunks)}")

    # Stage 1: summarize each chunk and save it immediately.
    for index, chunk in enumerate(chunks):
        chunk_path = chunks_dir / f"chunk_{index:03d}.md"

        if index in completed and chunk_path.exists():
            print(f"Skipping completed chunk {index + 1}/{len(chunks)}")
            continue

        print(f"Summarizing chunk {index + 1}/{len(chunks)}...")

        summary = summarize_text(
            chunk,
            instruction=(
                "Summarize this section of a research paper. "
                "Capture its key ideas, technical details, and findings. "
                "Keep the summary concise but informative."
            ),
        )

        chunk_path.write_text(summary + "\n", encoding="utf-8")

        completed.add(index)
        state["total_chunks"] = len(chunks)
        state["completed_chunks"] = sorted(completed)
        save_json(state_path, state)

        print(f"Saved: {chunk_path.name}")

    # Stage 2: combine the chunk summaries.
    summaries = [
        (chunks_dir / f"chunk_{i:03d}.md").read_text(encoding="utf-8")
        for i in range(len(chunks))
    ]

    combined = "\n\n".join(
        f"[Section {i + 1}]\n{summary}"
        for i, summary in enumerate(summaries)
    )

    print("Generating final summary...")

    final_summary = summarize_text(
        combined,
        instruction=(
            "Create one coherent final summary of the entire research paper. "
            "Combine related ideas, remove repetition, and preserve the paper's "
            "main problem, approach, key technical contributions, findings, "
            "and limitations when stated in the source. "
            "Do not claim that a result exists unless the source supports it."
        ),
    )

    final_path.write_text(final_summary + "\n", encoding="utf-8")

    print(f"Final summary saved to: {final_path}")
    print(final_summary)


if __name__ == "__main__":
    main()
