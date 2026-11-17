"""Unit tests for structure-aware document chunker using synthetic in-memory fixtures."""

from pathlib import Path
import sys
import unittest

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.chunking.chunker import DocumentChunk, chunk_document
from backend.document_parser.parser import ContentBlock, ParsedDocument, Section


class TestChunker(unittest.TestCase):
    """Test suite for structure-aware chunker."""

    def test_block_preservation_and_ordering(self):
        """All input blocks must be preserved across chunks in exact reading order."""
        b0 = ContentBlock(type="paragraph", text="Title & Authors", page=1, order=1)
        b1 = ContentBlock(type="paragraph", text="Intro paragraph 1", page=1, order=2)
        b2 = ContentBlock(type="paragraph", text="Intro paragraph 2", page=2, order=3)
        b3 = ContentBlock(type="paragraph", text="Model overview", page=2, order=4)
        b4 = ContentBlock(type="html_table", text="<table><tr><td>Arch</td></tr></table>", page=3, order=5)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=3,
            unassigned_blocks=[b0],
            sections=[
                Section(title="1 Introduction", level=1, page_start=1, page_end=2, content_blocks=[b1, b2]),
                Section(title="2 Architecture", level=1, page_start=2, page_end=3, content_blocks=[b3, b4]),
            ],
        )

        chunks = chunk_document(doc, target_size=100, max_size=200)

        # Collect all blocks across chunks
        preserved_blocks = []
        for chunk in chunks:
            preserved_blocks.extend(chunk.blocks)

        expected_blocks = [b0, b1, b2, b3, b4]
        self.assertEqual(preserved_blocks, expected_blocks)
        self.assertEqual([b.order for b in preserved_blocks], [1, 2, 3, 4, 5])

    def test_small_section_packing(self):
        """Multiple small sections should coalesce into a single chunk up to target_size."""
        b1 = ContentBlock(type="paragraph", text="Short intro content.", page=1, order=1)
        b2 = ContentBlock(type="paragraph", text="Short background content.", page=1, order=2)
        b3 = ContentBlock(type="paragraph", text="Short methods content.", page=2, order=3)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=2,
            sections=[
                Section(title="1 Intro", level=1, page_start=1, page_end=1, content_blocks=[b1]),
                Section(title="2 Background", level=1, page_start=1, page_end=1, content_blocks=[b2]),
                Section(title="3 Methods", level=1, page_start=2, page_end=2, content_blocks=[b3]),
            ],
        )

        # target_size = 500 is large enough for all 3 short sections
        chunks = chunk_document(doc, target_size=500, max_size=1000)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, 0)
        self.assertEqual(chunks[0].block_count, 3)
        self.assertIn("1 Intro", chunks[0].section_titles)
        self.assertIn("2 Background", chunks[0].section_titles)
        self.assertIn("3 Methods", chunks[0].section_titles)
        self.assertIn("## 1 Intro", chunks[0].text)
        self.assertIn("## 2 Background", chunks[0].text)
        self.assertIn("## 3 Methods", chunks[0].text)

    def test_oversized_section_and_continuation_headers(self):
        """Oversized sections must split along block boundaries with continuation headers."""
        b1 = ContentBlock(type="paragraph", text="A" * 600, page=2, order=1)
        b2 = ContentBlock(type="paragraph", text="B" * 600, page=3, order=2)
        b3 = ContentBlock(type="paragraph", text="C" * 600, page=4, order=3)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=4,
            sections=[
                Section(title="3 Model Architecture", level=1, page_start=2, page_end=4, content_blocks=[b1, b2, b3]),
            ],
        )

        # target_size = 800 forces each 600-char block into its own part
        chunks = chunk_document(doc, target_size=800, max_size=1500)

        self.assertEqual(len(chunks), 3)

        # Check part 1
        self.assertEqual(chunks[0].title, "3 Model Architecture (Part 1)")
        self.assertFalse(chunks[0].metadata["is_continuation"])
        self.assertIn("## 3 Model Architecture (Part 1)", chunks[0].text)
        self.assertEqual(chunks[0].blocks, [b1])

        # Check part 2
        self.assertEqual(chunks[1].title, "3 Model Architecture (Part 2)")
        self.assertTrue(chunks[1].metadata["is_continuation"])
        self.assertEqual(chunks[1].metadata["continuation_from"], "3 Model Architecture")
        self.assertIn("[Context: Continued from 3 Model Architecture", chunks[1].text)
        self.assertIn("## 3 Model Architecture (Part 2)", chunks[1].text)
        self.assertEqual(chunks[1].blocks, [b2])

        # Check part 3
        self.assertEqual(chunks[2].title, "3 Model Architecture (Part 3)")
        self.assertTrue(chunks[2].metadata["is_continuation"])
        self.assertEqual(chunks[2].blocks, [b3])

    def test_atomic_html_tables(self):
        """HTML tables must be preserved atomically without breaking HTML tags."""
        table_html = "<table>\n<tr><th>Model</th><th>BLEU</th></tr>\n<tr><td>Base</td><td>27.3</td></tr>\n</table>"
        b_para = ContentBlock(type="paragraph", text="Preceding paragraph description.", page=5, order=1)
        b_table = ContentBlock(type="html_table", text=table_html, page=5, order=2)
        b_after = ContentBlock(type="paragraph", text="Subsequent text analysis.", page=5, order=3)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=5,
            sections=[
                Section(title="Results", level=1, page_start=5, page_end=5, content_blocks=[b_para, b_table, b_after]),
            ],
        )

        chunks = chunk_document(doc, target_size=150, max_size=500)

        # Find the chunk containing the table
        table_chunks = [c for c in chunks if any(b.type == "html_table" for b in c.blocks)]
        self.assertEqual(len(table_chunks), 1)

        # Assert full intact table is present
        self.assertIn(table_html, table_chunks[0].text)
        self.assertIn("<table>", table_chunks[0].text)
        self.assertIn("</table>", table_chunks[0].text)

    def test_empty_sections_and_unassigned_blocks(self):
        """Empty sections must not yield empty chunks and unassigned blocks must be preserved."""
        b_front = ContentBlock(type="paragraph", text="Paper Title & Authors List", page=1, order=1)
        b_real = ContentBlock(type="paragraph", text="Real content for section 4.", page=2, order=2)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=2,
            unassigned_blocks=[b_front],
            sections=[
                # Empty sections as commonly found in OCR outputs
                Section(title="3.2.3", level=2, page_start=1, page_end=1, content_blocks=[]),
                Section(title="Applications of Attention", level=2, page_start=1, page_end=1, content_blocks=[]),
                Section(title="4 Conclusion", level=1, page_start=2, page_end=2, content_blocks=[b_real]),
            ],
        )

        chunks = chunk_document(doc, target_size=1000, max_size=2000)

        # Check that no chunk is empty
        for chunk in chunks:
            self.assertGreater(chunk.block_count, 0)
            self.assertGreater(len(chunk.text.strip()), 0)

        # Check unassigned blocks are preserved
        all_blocks = [b for c in chunks for b in c.blocks]
        self.assertIn(b_front, all_blocks)
        self.assertIn(b_real, all_blocks)

        # Check that empty section titles are carried forward into section_titles
        all_section_titles = [title for c in chunks for title in c.section_titles]
        self.assertIn("3.2.3", all_section_titles)
        self.assertIn("Applications of Attention", all_section_titles)
        self.assertIn("4 Conclusion", all_section_titles)

    def test_page_ranges_and_section_metadata(self):
        """Chunks must accurately report page ranges and section metadata."""
        b1 = ContentBlock(type="paragraph", text="Page 3 text", page=3, order=1)
        b2 = ContentBlock(type="paragraph", text="Page 4 text", page=4, order=2)
        b3 = ContentBlock(type="paragraph", text="Page 7 text", page=7, order=3)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=7,
            sections=[
                Section(title="Section A", level=1, page_start=3, page_end=7, content_blocks=[b1, b2, b3]),
            ],
        )

        chunks = chunk_document(doc, target_size=1000, max_size=2000)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]

        self.assertEqual(chunk.page_start, 3)
        self.assertEqual(chunk.page_end, 7)
        self.assertEqual(chunk.metadata["page_start"], 3)
        self.assertEqual(chunk.metadata["page_end"], 7)
        self.assertEqual(chunk.metadata["section_titles"], ["Section A"])
        self.assertEqual(chunk.block_count, 3)
        self.assertEqual(chunk.char_count, len(chunk.text))

    def test_oversized_individual_block_keep_atomic(self):
        """Under keep_atomic policy, an individual block > max_size is preserved without truncation."""
        huge_text = "D" * 3500
        b_huge = ContentBlock(type="paragraph", text=huge_text, page=1, order=1)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=1,
            sections=[
                Section(title="Big Block Sec", level=1, page_start=1, page_end=1, content_blocks=[b_huge]),
            ],
        )

        # max_size is 2000, smaller than block size 3500
        chunks = chunk_document(doc, target_size=1500, max_size=2000, oversized_policy="keep_atomic")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_count, 1)
        self.assertEqual(chunks[0].blocks[0], b_huge)
        # Verify 100% text preservation without silent truncation
        self.assertIn(huge_text, chunks[0].text)
        self.assertTrue(chunks[0].metadata.get("oversized"))
        self.assertEqual(chunks[0].metadata.get("policy"), "keep_atomic")

    def test_oversized_individual_block_split_policy(self):
        """Under split policy, an individual block > max_size is divided across sub-chunks without data loss."""
        part1 = "First segment of huge text.\n\n"
        part2 = "Second segment of huge text.\n\n"
        part3 = "Third segment of huge text."
        huge_text = part1 + part2 + part3
        b_huge = ContentBlock(type="paragraph", text=huge_text, page=2, order=1)

        doc = ParsedDocument(
            source="synthetic_test",
            pages=2,
            sections=[
                Section(title="Split Block Sec", level=1, page_start=2, page_end=2, content_blocks=[b_huge]),
            ],
        )

        chunks = chunk_document(doc, target_size=40, max_size=50, oversized_policy="split")

        self.assertGreater(len(chunks), 1)
        # All sub-chunks must respect max_size constraint for their block text
        for chunk in chunks:
            self.assertTrue(chunk.metadata.get("oversized_split"))

        # Reconstructed text from sub-chunks must contain all segments
        combined_text = "".join(b.text for c in chunks for b in c.blocks)
        self.assertEqual(combined_text, huge_text)

    def test_completely_empty_document(self):
        """An empty ParsedDocument must produce an empty list without errors."""
        doc = ParsedDocument(source="empty", pages=0)
        chunks = chunk_document(doc)
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
