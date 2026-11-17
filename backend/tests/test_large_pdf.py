
import argparse
import json
from pathlib import Path

import indra_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "large_pdf"
PAGES_DIR = OUTPUT_DIR / "pages"
STATE_PATH = OUTPUT_DIR / "state.json"


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))

    return {
        "source_pdf": None,
        "completed_pages": [],
    }


def save_state(state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(STATE_PATH)


def extract_pdf(pdf_path: Path, reset: bool = False):
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()

    if reset:
        for page_file in PAGES_DIR.glob("page_*.md"):
            page_file.unlink()

        state = {
            "source_pdf": str(pdf_path),
            "completed_pages": [],
        }

    elif state["source_pdf"] not in (None, str(pdf_path)):
        raise ValueError(
            "A different PDF was previously processed. "
            "Use --reset to start a new extraction."
        )

    state["source_pdf"] = str(pdf_path)
    save_state(state)

    completed = set(state["completed_pages"])

    print(f"PDF: {pdf_path}")
    print(f"Already completed: {len(completed)} pages")
    print("Loading Indra OCR...")

    converter = indra_ocr.load(threads=2)

    for page_number, page in converter.convert(
        str(pdf_path),
        max_pages=0,
    ):
        page_file = PAGES_DIR / f"page_{page_number:04d}.md"

        if page_number in completed and page_file.exists():
            continue

        page_text = (page.markdown or "").strip()

        # Save even empty pages so they are not processed repeatedly.
        page_file.write_text(page_text + "\n", encoding="utf-8")

        completed.add(page_number)
        state["completed_pages"] = sorted(completed)
        save_state(state)

        print(
            f"Saved page {page_number}: "
            f"{len(page_text)} characters"
        )

    print(f"\nExtraction finished.")
    print(f"Pages saved: {len(completed)}")
    print(f"Output directory: {PAGES_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all PDF pages with resumable page files."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete saved page files and start extraction again",
    )
    args = parser.parse_args()

    extract_pdf(args.pdf.expanduser().resolve(), reset=args.reset)


if __name__ == "__main__":
    main()
