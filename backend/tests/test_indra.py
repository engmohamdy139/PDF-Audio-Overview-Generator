
import sys
from pathlib import Path

import indra_ocr


def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/tests/test_indra.py path/to/file.pdf")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()

    if not pdf_path.is_file():
        print(f"File not found: {pdf_path}")
        raise SystemExit(1)

    print("Loading Indra OCR...")
    conv = indra_ocr.load(threads=2)

    print(f"Processing: {pdf_path.name}")

    page_count = 0

    for page_number, page in conv.convert(str(pdf_path), max_pages=3):
        page_count += 1
        print(f"\n--- Page {page_number} ---")
        print(page.markdown)

    if page_count == 0:
        print("No pages were returned.")
    else:
        print(f"\nFinished. Pages processed: {page_count}")


if __name__ == "__main__":
    main()
