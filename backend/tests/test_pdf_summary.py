
import sys
from pathlib import Path

import indra_ocr
import requests


PDF_PATH = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "~/Downloads/Attention is all you need.pdf"
).expanduser().resolve()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b-instruct"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from the first three PDF pages."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Loading Indra OCR...")
    converter = indra_ocr.load(threads=2)

    pages = []

    print(f"Extracting text from: {pdf_path.name}")

    for page_number, page in converter.convert(
        str(pdf_path), max_pages=3
    ):
        page_text = page.markdown or ""
        pages.append(f"\n--- Page {page_number} ---\n{page_text}")
        print(f"Extracted page {page_number}: {len(page_text)} characters")

    text = "\n".join(pages).strip()

    if not text:
        raise ValueError("No text was extracted from the PDF.")

    return text


def summarize_text(text: str) -> str:
    """Summarize extracted text using the local Ollama model."""
    prompt = f"""
You are an assistant helping readers understand research papers.

Summarize the provided text in clear, natural English.
Explain the main problem, the proposed approach, and the key ideas.
Use only information supported by the text.
Do not invent results or details.
If the text is incomplete, mention that the summary covers only an excerpt.

PAPER EXCERPT:
{text}

Write a concise summary with clear headings.
"""

    print("\nSending extracted text to Qwen...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
            },
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    text = extract_pdf_text(PDF_PATH)

    print(f"\nTotal extracted characters: {len(text)}")

    summary = summarize_text(text)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "summary.md"
    output_file.write_text(summary, encoding="utf-8")

    print("\n" + "=" * 50)
    print("GENERATED SUMMARY")
    print("=" * 50)
    print(summary)
    print(f"\nSummary saved to: {output_file.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"\nOllama API error: {exc}")
        print("Check that Ollama is running and the model is available.")
        raise SystemExit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        raise SystemExit(1)
