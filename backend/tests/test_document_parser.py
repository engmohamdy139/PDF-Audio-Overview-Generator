import re
from pathlib import Path
import sys
import unittest

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from backend.document_parser.parser import parse_markdown_pages


PAGES_DIR = BASE_DIR / "outputs" / "large_pdf" / "pages"


class TestDocumentParser(unittest.TestCase):

    def test_large_pdf_structure(self):
        document = parse_markdown_pages(PAGES_DIR)

        titles = [section.title for section in document.sections]

        self.assertEqual(document.pages, 15)
        self.assertIn("1 Introduction", titles)
        self.assertIn("2 Background", titles)
        self.assertIn("3 Model Architecture", titles)

        self.assertEqual(len(titles), len(set(titles)))

    def test_section_order(self):
        document = parse_markdown_pages(PAGES_DIR)
        titles = [section.title for section in document.sections]

        self.assertLess(
            titles.index("1 Introduction"),
            titles.index("2 Background"),
        )

        self.assertLess(
            titles.index("2 Background"),
            titles.index("3 Model Architecture"),
        )

    def test_content_and_page_metadata(self):
        document = parse_markdown_pages(PAGES_DIR)

        introduction = next(
            section for section in document.sections
            if section.title == "1 Introduction"
        )

        self.assertGreater(len(introduction.content_blocks), 0)
        self.assertEqual(introduction.page_start, 2)

        for section in document.sections:
            self.assertGreaterEqual(section.page_end, section.page_start)

            for block in section.content_blocks:
                self.assertGreaterEqual(block.page, 1)
                self.assertLessEqual(block.page, document.pages)
    def test_html_tables_preserved_exactly(self):
        """Every HTML table must be preserved exactly as in its source page."""
        for page_no in [3, 4, 14, 15]:
            source_path = (
                PAGES_DIR / f"page_{page_no - 1:04d}.md"
            )
            source = source_path.read_text(encoding="utf-8")

            source_tables = re.findall(
                r"<table>.*?</table>",
                source,
                flags=re.DOTALL,
            )

            document = parse_markdown_pages(PAGES_DIR)
            parsed_tables = [
                block.text
                for block in document.content_blocks
                if block.page == page_no
                and block.type == "html_table"
            ]

            self.assertEqual(
                parsed_tables,
                source_tables,
                f"HTML table content changed on page {page_no}",
            )

    def test_reference_pages_preserved(self):
        """Reference text on pages 11 and 12 must not be lost."""
        document = parse_markdown_pages(PAGES_DIR)

        for page_no in [11, 12]:
            source_path = (
                PAGES_DIR / f"page_{page_no - 1:04d}.md"
            )
            source = source_path.read_text(encoding="utf-8").strip()

            parsed_text = "\n".join(
                block.text
                for block in document.content_blocks
                if block.page == page_no
            )

            self.assertIn(
                source,
                parsed_text,
                f"Reference text missing on page {page_no}",
            )

    def test_paragraphs_do_not_cross_page_boundaries(self):
        """Text from separate page files must retain its page number."""
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            pages_dir = Path(temp_dir)

            (pages_dir / "page_0000.md").write_text(
                "First page text\n",
                encoding="utf-8",
            )
            (pages_dir / "page_0001.md").write_text(
                "Second page text\n",
                encoding="utf-8",
            )

            document = parse_markdown_pages(pages_dir)

            page_one_blocks = [
                block for block in document.content_blocks
                if block.page == 1
            ]
            page_two_blocks = [
                block for block in document.content_blocks
                if block.page == 2
            ]

            self.assertTrue(
                any("First page text" in b.text for b in page_one_blocks)
            )
            self.assertTrue(
                any("Second page text" in b.text for b in page_two_blocks)
            )
            self.assertFalse(
                any("Second page text" in b.text for b in page_one_blocks)
            )

if __name__ == "__main__":
    unittest.main()
