"""Structure-aware document chunker for the PDF Audio Generator.

Consumes ParsedDocument instances directly from the document parser,
coalesces small sections, splits oversized sections between content blocks,
preserves HTML tables and equations atomically, and avoids character-level
slicing and overlap.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from backend.document_parser.parser import ContentBlock, ParsedDocument, Section


@dataclass
class DocumentChunk:
    """Reusable chunk representation ready for downstream LLM consumption."""

    chunk_id: int
    title: str
    section_titles: list[str]
    text: str
    page_start: int
    page_end: int
    block_count: int
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: list[ContentBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a dictionary."""
        return asdict(self)


@dataclass
class _SectionUnit:
    """Internal representation of a section or front-matter unit to chunk."""

    title: str
    level: int
    blocks: list[ContentBlock]
    is_unassigned: bool = False


def _split_text_into_subblocks(text: str, max_size: int) -> list[str]:
    """Split oversized text into parts <= max_size without character or delimiter loss."""
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_size:
            parts.append(text[start:])
            break

        candidate_end = start + max_size
        split_pos = -1

        # Prefer paragraph break
        p_idx = text.rfind("\n\n", start, candidate_end)
        if p_idx != -1 and p_idx > start:
            split_pos = p_idx + 2
        else:
            # Fallback to line break
            l_idx = text.rfind("\n", start, candidate_end)
            if l_idx != -1 and l_idx > start:
                split_pos = l_idx + 1
            else:
                # Fallback to sentence break
                sentences = list(re.finditer(r"[.!?]\s+", text[start:candidate_end]))
                if sentences:
                    split_pos = start + sentences[-1].end()

        # Hard fallback to max_size
        if split_pos == -1 or split_pos <= start:
            split_pos = candidate_end

        parts.append(text[start:split_pos])
        start = split_pos

    return parts


def _format_chunk_text(
    section_groups: list[tuple[str, list[ContentBlock]]],
    chunk_title: str = "",
    is_continuation: bool = False,
    continuation_from: str = "",
    continuation_page: int = 1,
) -> str:
    """Format section titles, continuation breadcrumbs, and blocks into Markdown text."""
    parts: list[str] = []

    if is_continuation:
        from_title = continuation_from or chunk_title or "Previous Section"
        parts.append(f"[Context: Continued from {from_title}, Page {continuation_page}]")

    for idx, (title, blocks) in enumerate(section_groups):
        heading_title = chunk_title if (len(section_groups) == 1 and chunk_title) else title
        if heading_title:
            parts.append(f"## {heading_title}")

        for block in blocks:
            parts.append(block.text)

    return "\n\n".join(parts)


def chunk_document(
    document: ParsedDocument,
    target_size: int = 5000,
    max_size: int = 6000,
    oversized_policy: str = "keep_atomic",
) -> list[DocumentChunk]:
    """Convert a ParsedDocument into structure-aware, prompt-sized DocumentChunks.

    Args:
        document: The ParsedDocument produced by the document parser.
        target_size: Soft character threshold to pack small sections or split large ones.
        max_size: Hard character threshold for triggering oversized block handling.
        oversized_policy: Policy for ContentBlock instances exceeding max_size.
            - "keep_atomic" (default): Keep the oversized block completely intact in its
              own dedicated chunk without silent truncation. Sets metadata['oversized'] = True.
            - "split": Divide the oversized block text across multiple continuation chunks
              at paragraph, line, or sentence boundaries without data loss.

    Returns:
        A list of DocumentChunk objects covering all blocks in reading order.
    """
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer.")
    if max_size < target_size:
        raise ValueError("max_size cannot be smaller than target_size.")
    if oversized_policy not in ("keep_atomic", "split"):
        raise ValueError(f"Unknown oversized_policy: {oversized_policy}. Expected 'keep_atomic' or 'split'.")

    # 1. Normalize document content into a stream of SectionUnits
    units: list[_SectionUnit] = []

    if document.unassigned_blocks:
        units.append(
            _SectionUnit(
                title="Front Matter",
                level=1,
                blocks=list(document.unassigned_blocks),
                is_unassigned=True,
            )
        )

    if document.sections:
        for s in document.sections:
            units.append(
                _SectionUnit(
                    title=s.title,
                    level=s.level,
                    blocks=list(s.content_blocks),
                )
            )
    elif not document.unassigned_blocks and document.content_blocks:
        # Fallback when sections are not structured but content_blocks exist
        units.append(
            _SectionUnit(
                title="Document",
                level=1,
                blocks=list(document.content_blocks),
            )
        )

    if not units:
        return []

    # 2. Process units, carrying forward empty section titles
    chunks: list[DocumentChunk] = []
    chunk_counter = 0
    pending_empty_titles: list[str] = []

    # Accumulator for coalescing small sections
    current_groups: list[tuple[str, list[ContentBlock]]] = []
    current_titles: list[str] = []
    current_blocks: list[ContentBlock] = []
    current_char_count = 0

    def flush_current_chunk(
        chunk_title: str | None = None,
        is_continuation: bool = False,
        continuation_from: str = "",
        part_index: int = 1,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        nonlocal chunk_counter, current_groups, current_titles, current_blocks, current_char_count

        if not current_blocks:
            return

        page_start = min(b.page for b in current_blocks)
        page_end = max(b.page for b in current_blocks)
        title = chunk_title or (current_titles[0] if current_titles else "Section")

        formatted_text = _format_chunk_text(
            section_groups=current_groups,
            chunk_title=title,
            is_continuation=is_continuation,
            continuation_from=continuation_from,
            continuation_page=page_start,
        )

        metadata: dict[str, Any] = {
            "page_start": page_start,
            "page_end": page_end,
            "section_titles": list(current_titles),
            "is_continuation": is_continuation,
        }
        if is_continuation:
            metadata["continuation_from"] = continuation_from
            metadata["part"] = part_index
        if extra_metadata:
            metadata.update(extra_metadata)

        chunk = DocumentChunk(
            chunk_id=chunk_counter,
            title=title,
            section_titles=list(current_titles),
            text=formatted_text,
            page_start=page_start,
            page_end=page_end,
            block_count=len(current_blocks),
            char_count=len(formatted_text),
            metadata=metadata,
            blocks=list(current_blocks),
        )
        chunks.append(chunk)
        chunk_counter += 1

        # Reset accumulator
        current_groups = []
        current_titles = []
        current_blocks = []
        current_char_count = 0

    for unit in units:
        if not unit.blocks:
            # Empty section: preserve its title for the next non-empty section
            pending_empty_titles.append(unit.title)
            continue

        unit_titles = pending_empty_titles + [unit.title]
        pending_empty_titles = []

        unit_chars = sum(len(b.text) for b in unit.blocks)

        # Check if unit fits into current accumulator (Small Section Coalescing)
        if current_blocks and (current_char_count + unit_chars <= target_size):
            current_groups.append((unit.title, unit.blocks))
            current_titles.extend(unit_titles)
            current_blocks.extend(unit.blocks)
            current_char_count += unit_chars
            continue

        # If it doesn't fit, flush existing accumulated content
        if current_blocks:
            flush_current_chunk()

        # Now processing unit with a fresh accumulator.
        # Check if unit fits in a single chunk
        if unit_chars <= target_size:
            current_groups.append((unit.title, unit.blocks))
            current_titles.extend(unit_titles)
            current_blocks.extend(unit.blocks)
            current_char_count = unit_chars
            continue

        # Unit is oversized (> target_size). Split across block boundaries.
        part_index = 1
        part_blocks: list[ContentBlock] = []
        part_chars = 0
        section_titles_for_first_part = list(unit_titles)

        for block in unit.blocks:
            block_len = len(block.text)

            # Check for individual oversized block
            if block_len > max_size:
                # Flush any accumulated blocks in this part before the oversized block
                if part_blocks:
                    current_groups = [(unit.title, part_blocks)]
                    current_titles = section_titles_for_first_part if part_index == 1 else [unit.title]
                    current_blocks = list(part_blocks)
                    flush_current_chunk(
                        chunk_title=f"{unit.title} (Part {part_index})",
                        is_continuation=(part_index > 1),
                        continuation_from=unit.title,
                        part_index=part_index,
                    )
                    part_index += 1
                    part_blocks = []
                    part_chars = 0

                # Handle oversized block based on policy
                if oversized_policy == "keep_atomic":
                    # Place oversized block into its own dedicated chunk without truncation
                    current_groups = [(unit.title, [block])]
                    current_titles = section_titles_for_first_part if part_index == 1 else [unit.title]
                    current_blocks = [block]
                    flush_current_chunk(
                        chunk_title=f"{unit.title} (Part {part_index})",
                        is_continuation=(part_index > 1),
                        continuation_from=unit.title,
                        part_index=part_index,
                        extra_metadata={
                            "oversized": True,
                            "policy": "keep_atomic",
                        },
                    )
                    part_index += 1
                else:
                    # Policy: split text into continuation sub-blocks
                    sub_texts = _split_text_into_subblocks(block.text, max_size)
                    for sub_text in sub_texts:
                        sub_block = ContentBlock(
                            type=block.type,
                            text=sub_text,
                            page=block.page,
                            order=block.order,
                            needs_review=block.needs_review,
                        )
                        current_groups = [(unit.title, [sub_block])]
                        current_titles = section_titles_for_first_part if part_index == 1 else [unit.title]
                        current_blocks = [sub_block]
                        flush_current_chunk(
                            chunk_title=f"{unit.title} (Part {part_index})",
                            is_continuation=(part_index > 1),
                            continuation_from=unit.title,
                            part_index=part_index,
                            extra_metadata={
                                "oversized_split": True,
                                "policy": "split",
                            },
                        )
                        part_index += 1
                continue

            # Standard block: check if adding it exceeds target_size
            if part_blocks and (part_chars + block_len > target_size):
                current_groups = [(unit.title, part_blocks)]
                current_titles = section_titles_for_first_part if part_index == 1 else [unit.title]
                current_blocks = list(part_blocks)
                flush_current_chunk(
                    chunk_title=f"{unit.title} (Part {part_index})",
                    is_continuation=(part_index > 1),
                    continuation_from=unit.title,
                    part_index=part_index,
                )
                part_index += 1
                part_blocks = [block]
                part_chars = block_len
            else:
                part_blocks.append(block)
                part_chars += block_len

        # Remaining blocks of this oversized section
        if part_blocks:
            current_groups = [(unit.title, part_blocks)]
            current_titles = section_titles_for_first_part if part_index == 1 else [unit.title]
            current_blocks = list(part_blocks)
            flush_current_chunk(
                chunk_title=f"{unit.title} (Part {part_index})",
                is_continuation=(part_index > 1),
                continuation_from=unit.title,
                part_index=part_index,
            )

    # Flush any remaining accumulated blocks at end of document
    if current_blocks:
        flush_current_chunk()

    # If there are trailing empty section titles and we produced chunks, attach them to the last chunk
    if pending_empty_titles and chunks:
        chunks[-1].section_titles.extend(pending_empty_titles)
        chunks[-1].metadata.setdefault("trailing_empty_sections", []).extend(pending_empty_titles)

    return chunks
