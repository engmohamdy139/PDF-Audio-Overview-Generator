
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED_SECTION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*\.?)\s+([A-Z][^\n]{1,100})$"
)
INCOMPLETE_NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)*\.?$")


@dataclass
class ContentBlock:
    type: str
    text: str
    page: int
    order: int = 0
    needs_review: bool = False


@dataclass
class Section:
    title: str
    level: int
    page_start: int
    page_end: int
    content_blocks: list[ContentBlock] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    needs_review: bool = False


@dataclass
class ParsedDocument:
    source: str
    pages: int
    content_blocks: list[ContentBlock] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    unassigned_blocks: list[ContentBlock] = field(default_factory=list)


def heading_review_reason(text: str) -> str | None:
    """Return a review reason for suspicious or incomplete headings."""
    text = text.strip()

    if INCOMPLETE_NUMBER_PATTERN.fullmatch(text):
        return "incomplete_numbered_heading"

    if len(text) > 150:
        return "heading_too_long"

    if any(symbol in text for symbol in ("=", "→", "∑", "√")):
        return "possible_formula_heading"

    if re.search(r"\b(Q|K|V)\s*,", text):
        return "possible_formula_heading"

    if re.search(r"\b(?:p|y)\s+p\s+y\b", text, re.IGNORECASE):
        return "possible_ocr_corruption"

    return None


def is_likely_heading(text: str, markdown_level: int) -> bool:
    """Conservative first-pass heading validation."""
    text = text.strip()

    if not text or len(text) > 150:
        return False

    # OCR may incorrectly mark equations as Markdown headings.
    if any(symbol in text for symbol in ("=", "→", "∑", "√")):
        return False

    if re.search(r"\b(Q|K|V)\s*,", text):
        return False

    # Markdown heading is useful evidence, but not sufficient alone.
    # Numbered sections and short title-like headings are accepted.
    if NUMBERED_SECTION_PATTERN.match(text):
        return True

    if markdown_level >= 2 and len(text.split()) <= 12:
        return True

    return False

def parse_markdown_pages(pages_dir: Path) -> ParsedDocument:
    files = sorted(pages_dir.glob("page_*.md"))

    if not files:
        raise FileNotFoundError(
            f"No page Markdown files found in: {pages_dir}"
        )

    document = ParsedDocument(
        source=str(pages_dir),
        pages=len(files),
    )

    current_section: Section | None = None
    paragraph_lines: list[str] = []
    paragraph_page: int | None = None
    order = 0

    def add_block(block: ContentBlock) -> None:
        """Keep every parsed block in document order and assign its section."""
        document.content_blocks.append(block)

        if current_section is None:
            document.unassigned_blocks.append(block)
        else:
            current_section.content_blocks.append(block)
            current_section.page_end = max(
                current_section.page_end, block.page
            )

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_page, order

        if not paragraph_lines or paragraph_page is None:
            paragraph_lines = []
            paragraph_page = None
            return

        text = "\n".join(paragraph_lines).strip()

        if text:
            order += 1
            add_block(
                ContentBlock(
                    type="paragraph",
                    text=text,
                    page=paragraph_page,
                    order=order,
                )
            )

        paragraph_lines = []
        paragraph_page = None

    for file in files:
        match = re.search(r"page_(\d+)\.md$", file.name)
        page_number = int(match.group(1)) + 1 if match else 0

        # Finish the previous page's paragraph before moving to this page.
        flush_paragraph()

        content = file.read_text(encoding="utf-8")
        lines = content.splitlines()
        line_index = 0

        while line_index < len(lines):
            line = lines[line_index]
            line_index += 1
            stripped = line.strip()

            # Preserve complete HTML tables as single blocks.
            if stripped.lower().startswith("<table"):
                flush_paragraph()
                html_lines = [line]

                while (
                    "</table>" not in html_lines[-1].lower()
                    and line_index < len(lines)
                ):
                    html_lines.append(lines[line_index])
                    line_index += 1

                order += 1
                add_block(
                    ContentBlock(
                        type="html_table",
                        text="\n".join(html_lines),
                        page=page_number,
                        order=order,
                    )
                )
                continue

            if not stripped:
                flush_paragraph()
                continue

            heading_match = HEADING_PATTERN.match(stripped)

            if heading_match:
                flush_paragraph()

                markdown_level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                if is_likely_heading(title, markdown_level):
                    order += 1
                    reason = heading_review_reason(title)

                    section = Section(
                        title=title,
                        level=markdown_level - 1,
                        page_start=page_number,
                        page_end=page_number,
                        needs_review=reason is not None,
                    )
                    document.sections.append(section)
                    current_section = section

                    # Headings are content too; preserve them in the main stream.
                    document.content_blocks.append(
                        ContentBlock(
                            type="heading",
                            text=title,
                            page=page_number,
                            order=order,
                            needs_review=reason is not None,
                        )
                    )
                    continue

                # Keep rejected heading-like lines as text instead of dropping them.
                stripped = title

            if stripped.startswith("|"):
                flush_paragraph()
                order += 1

                add_block(
                    ContentBlock(
                        type="table_row",
                        text=stripped,
                        page=page_number,
                        order=order,
                    )
                )
                continue

            if paragraph_page is None:
                paragraph_page = page_number

            paragraph_lines.append(stripped)

    flush_paragraph()

    return document


def document_to_dict(document: ParsedDocument) -> dict:
    return asdict(document)


def save_document(document: ParsedDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            document_to_dict(document),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

