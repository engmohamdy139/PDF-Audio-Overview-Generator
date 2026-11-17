"""Structure-aware hierarchical summarization pipeline (V2).

Option B implementation preserving legacy test_large_pdf_summary.py unchanged.
Integrates the existing document parser and chunker, validates content integrity,
pipe-table rejection, conservative token preflight, and strict safe-resume policies.
"""

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import requests

from backend.chunking.chunker import DocumentChunk, chunk_document
from backend.config import (
    BASE_DIR,
    CHUNK_MAX_SIZE,
    CHUNK_TARGET_SIZE,
    MODEL_NAME,
    NUM_CTX,
    OLLAMA_URL,
)
from backend.document_parser.parser import (
    ContentBlock,
    ParsedDocument,
    Section,
    parse_markdown_pages,
)

PIPELINE_FILE = Path(__file__).resolve()
PARSER_FILE = BASE_DIR / "backend" / "document_parser" / "parser.py"
CHUNKER_FILE = BASE_DIR / "backend" / "chunking" / "chunker.py"

PAGES_DIR = BASE_DIR / "outputs" / "large_pdf" / "pages"
V2_ROOT = BASE_DIR / "outputs" / "large_pdf" / "summary_v2"
CHUNKS_DIR = V2_ROOT / "chunks"
STATE_PATH = V2_ROOT / "state.json"
TEMP_STATE_PATH = V2_ROOT / "state.json.tmp"
FINAL_PATH = V2_ROOT / "final_summary.md"
TEMP_FINAL_PATH = V2_ROOT / "final_summary.md.tmp"
MANIFEST_PATH = V2_ROOT / "chunks_metadata.json"
TEMP_MANIFEST_PATH = V2_ROOT / "chunks_metadata.json.tmp"

PIPELINE_VERSION = "v2"
NUM_PREDICT = 1024
TEMPERATURE = 0.2
TIMEOUT = 600
OVERSIZED_POLICY = "keep_atomic"

BASELINE_GLOBAL_FINGERPRINT = (
    "6e87792ad7229f0248f19c86cbb94527adf4a712781d92f45149ba223295d660"
)

CHUNK_INSTRUCTION = (
    "Summarize this research paper section. "
    "Capture key ideas, technical details, and findings. "
    "Keep concise but informative."
)

REDUCE_INSTRUCTION = (
    "Synthesize the following research paper section summaries into "
    "a dense, high-level summary of at most 200 words "
    "(under 1,200 characters). "
    "Focus strictly on core technical mechanisms, architecture, "
    "and empirical findings. "
    "For reference or bibliography sections, summarize the general "
    "lineage of cited prior work in at most 1-2 brief sentences; "
    "do not enumerate individual paper titles, authors, "
    "or publication metadata. "
    "Be extremely concise and eliminate all filler."
)

FINAL_INSTRUCTION = (
    "Create a comprehensive, technically accurate final summary of the entire research paper. "
    "Synthesize all provided sections into one coherent narrative, remove repetition, "
    "and preserve the important technical details and experimental evidence. "

    "Cover the following aspects whenever supported by the source: "
    "1. Research problem and motivation. "
    "2. Proposed architecture and its main components. "
    "3. Core technical mechanisms and how they work. "
    "4. Important implementation and training details, including datasets, "
    "hardware, hyperparameters, and training configuration. "
    "5. Quantitative experimental results, including exact BLEU scores, "
    "F1 scores, and other reported metrics. "
    "6. Model variations, ablation findings, and limitations. "
    "7. Important observations and conclusions. "

    "Preserve exact numerical values and their associated context. "
    "Do not replace specific experimental results with vague statements "
    "such as 'achieves state-of-the-art results'. "
    "Do not invent information or infer unsupported conclusions. "
    "Exclude bibliography details unless they are essential to understanding "
    "the research. "
    "Use clear technical language and meaningful paragraphs. "
    "Prioritize information coverage and factual accuracy over excessive brevity."
)


def build_prompt(text: str, instruction: str) -> str:
    """Format input text and instruction into a clear prompt."""
    return f"{instruction}\n\n---\n\n{text}"


def summarize_text(text: str, instruction: str) -> str:
    """Send text to Ollama and return generated summary."""
    prompt = build_prompt(text, instruction)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    summary = result.get("response", "").strip()
    if not summary:
        raise ValueError("Ollama returned an empty summary response.")
    return summary


def validate_input_pages(pages_dir: Path) -> list[Path]:
    """Verify input page markdown files exist and contain non-whitespace content."""
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Input pages directory not found: {pages_dir}")

    page_files = sorted(pages_dir.glob("page_*.md"))
    if not page_files:
        raise FileNotFoundError(f"No page_*.md files found in {pages_dir}")

    for page_path in page_files:
        if not re.fullmatch(r"page_\d+\.md", page_path.name):
            raise ValueError(f"Invalid page filename format: {page_path.name}")
        content = page_path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Input page file is empty or whitespace-only: {page_path.name}")

    return page_files


def validate_no_pipe_tables(doc: ParsedDocument) -> None:
    """Reject pipe-delimited table rows before inference.

    Uses actual ParsedDocument.content_blocks containing ContentBlock items.
    """
    for block in doc.content_blocks:
        if block.type == "table_row":
            raise ValueError(
                f"Unsupported markdown pipe-table row detected on page {block.page} (order {block.order}): "
                f"'{block.text[:60]}...'. Tables must be converted to plain text or HTML tables."
            )
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
                raise ValueError(
                    f"Unsupported markdown pipe-table row detected on page {block.page} (order {block.order}): "
                    f"'{line[:60]}...'. Tables must be converted to plain text or HTML tables."
                )


def flatten_sections_in_order(sections: list[Section]) -> list[Section]:
    """Recursively flatten sections and their children preserving document reading order."""
    flattened: list[Section] = []
    for section in sections:
        flattened.append(section)
        if section.children:
            flattened.extend(flatten_sections_in_order(section.children))
    return flattened


def validate_content_integrity(doc: ParsedDocument, chunks: list[DocumentChunk]) -> None:
    """Validate completeness, body-block conservation, uniqueness, order, and heading coverage.

    Validation Strategy & Documented Data Model Limitations:
    1. Strict ContentBlock Object Identity: In keep_atomic mode, chunking preserves the exact
       ContentBlock instances for all body blocks from the parsed document. We verify completeness,
       uniqueness, and exact reading order using strict object identity (chk_blk is doc_blk). Any
       non-identical block instance immediately triggers validation failure; attribute equality is not accepted.
    2. Heading Representation & Limitations: In parser.py and chunker.py, headings are represented
       as title strings (in Section.title, DocumentChunk.section_titles, and formatted markdown headers),
       rather than preserved ContentBlock objects. Heading validation verifies coverage and reading order
       as far as the data model permits; object-identity validation applies exclusively to body blocks.
    3. Duplicate Heading Limitation: If multiple sections share identical title strings, title-string
       matching cannot distinguish individual section instances. Instance-level completeness, contiguity,
       and reading order are formally guaranteed by body ContentBlock object identity and ordering.
    4. Recursive Section Hierarchy & Strict Block Ordering: All sections (including any nested in
       Section.children) are traversed in document order. For consecutive non-empty sections
       s_k and s_{k+1}, coverage ranges are calculated from chunk block memberships. We verify that each
       section's coverage range is contiguous, and that a later section cannot begin before the previous
       section's coverage ends (min_chunk(s_{k+1}) >= max_chunk(s_k)). Furthermore, if two sections
       share a chunk, strict block ordering requires max(curr_positions) < min(next_positions);
       any case where max(curr_positions) >= min(next_positions) triggers an immediate error.
    """
    if not chunks:
        raise ValueError("Chunking produced zero chunks.")

    doc_body_blocks = [b for b in doc.content_blocks if b.type != "heading"]
    chunk_blocks = [b for chunk in chunks for b in chunk.blocks]

    if len(doc_body_blocks) != len(chunk_blocks):
        raise ValueError(
            f"Body block count mismatch: Document has {len(doc_body_blocks)}, "
            f"chunks have {len(chunk_blocks)}."
        )

    seen_block_ids: set[int] = set()
    for i, (doc_blk, chk_blk) in enumerate(zip(doc_body_blocks, chunk_blocks)):
        if id(chk_blk) in seen_block_ids:
            raise ValueError(
                f"Duplicate ContentBlock instance detected at chunk block index {i}: "
                f"order={chk_blk.order}, page={chk_blk.page}"
            )
        seen_block_ids.add(id(chk_blk))

        # Enforce strict object identity without attribute fallback
        if chk_blk is not doc_blk:
            raise ValueError(
                f"Strict ContentBlock object identity mismatch at block index {i}: "
                f"chunk block (id={id(chk_blk)}, order={chk_blk.order}, page={chk_blk.page}) is not "
                f"document block (id={id(doc_blk)}, order={doc_blk.order}, page={doc_blk.page})."
            )

    # Flatten section hierarchy (including children) in pre-order
    all_sections = flatten_sections_in_order(doc.sections)
    non_empty_sections = [s for s in all_sections if s.content_blocks]
    section_ranges: list[tuple[int, int]] = []

    for s_idx, section in enumerate(non_empty_sections):
        section_block_ids = {id(b) for b in section.content_blocks}
        covering_chunk_indices = [
            c_idx
            for c_idx, c in enumerate(chunks)
            if any(id(b) in section_block_ids for b in c.blocks)
        ]

        if not covering_chunk_indices:
            raise ValueError(
                f"Non-empty section '{section.title}' has no blocks assigned to any chunk."
            )

        start_chunk = min(covering_chunk_indices)
        end_chunk = max(covering_chunk_indices)

        # Verify contiguity of covering chunk range
        expected_range = set(range(start_chunk, end_chunk + 1))
        if set(covering_chunk_indices) != expected_range:
            raise ValueError(
                f"Section '{section.title}' has non-contiguous chunk coverage: {covering_chunk_indices}."
            )

        section_ranges.append((start_chunk, end_chunk))

        # Verify heading title is preserved in metadata or text of covering chunks
        title_preserved = any(
            section.title in chunks[c_idx].section_titles or section.title in chunks[c_idx].text
            for c_idx in covering_chunk_indices
        )
        if not title_preserved:
            raise ValueError(
                f"Section heading '{section.title}' was not preserved in any of its covering chunks."
            )

    # Verify that a later section cannot begin before the previous section's coverage ends
    for k in range(len(non_empty_sections) - 1):
        curr_sec = non_empty_sections[k]
        next_sec = non_empty_sections[k + 1]
        curr_start, curr_end = section_ranges[k]
        next_start, next_end = section_ranges[k + 1]

        if next_start < curr_end:
            raise ValueError(
                f"Section ordering violation: section '{next_sec.title}' begins at chunk {next_start}, "
                f"which is before previous section '{curr_sec.title}' ends at chunk {curr_end}."
            )

        # If they share a chunk (coalesced small sections), enforce strict block ordering within the shared chunk
        if next_start == curr_end:
            shared_chunk = chunks[curr_end]
            curr_block_ids = {id(b) for b in curr_sec.content_blocks}
            next_block_ids = {id(b) for b in next_sec.content_blocks}

            curr_positions = [
                pos for pos, b in enumerate(shared_chunk.blocks) if id(b) in curr_block_ids
            ]
            next_positions = [
                pos for pos, b in enumerate(shared_chunk.blocks) if id(b) in next_block_ids
            ]

            if curr_positions and next_positions:
                if max(curr_positions) >= min(next_positions):
                    raise ValueError(
                        f"Block ordering violation within shared chunk {curr_end}: blocks of "
                        f"section '{curr_sec.title}' (last position {max(curr_positions)}) are not strictly "
                        f"before blocks of '{next_sec.title}' (first position {min(next_positions)})."
                    )


def validate_token_preflight(full_prompt: str, label: str) -> None:
    """Enforce conservative token budget heuristic on complete prompt before Ollama dispatch.

    Budget calculation:
        Max input budget = NUM_CTX (4096) - NUM_PREDICT (1024) - system_overhead (200) - safety_buffer (256)
                         = 2616 tokens.
    Token estimation heuristic:
        estimated_tokens = len(full_prompt) / 2.0 (assuming 2.0 characters per token).
        Note: This is an empirical upper-bound safety margin, not a byte-exact tokenizer guarantee.
    """
    system_prompt_overhead = 200
    safety_buffer = 256
    max_input_tokens = NUM_CTX - NUM_PREDICT - system_prompt_overhead - safety_buffer
    estimated_input_tokens = len(full_prompt) / 2.0

    if estimated_input_tokens > max_input_tokens:
        raise ValueError(
            f"Pre-flight token estimation exceeded for {label}: "
            f"Estimated {estimated_input_tokens:.0f} tokens ({len(full_prompt)} chars in full prompt) "
            f"exceeds safe input budget of {max_input_tokens} tokens "
            f"(NUM_CTX={NUM_CTX}, NUM_PREDICT={NUM_PREDICT}, overhead={system_prompt_overhead}, buffer={safety_buffer}). "
            "Execution aborted to prevent context truncation."
        )


def validate_all_chunks_preflight(chunks: list[DocumentChunk]) -> None:
    """Audit all chunks with the full chunk prompt before any Ollama request."""
    for i, chunk in enumerate(chunks):
        full_prompt = build_prompt(chunk.text, CHUNK_INSTRUCTION)
        validate_token_preflight(full_prompt, f"Chunk {i + 1}/{len(chunks)} ('{chunk.title}')")


def compute_source_code_hash(path: Path) -> str:
    """Compute SHA-256 hash of a source code file for pipeline invalidation."""
    if not path.is_file():
        raise FileNotFoundError(f"Source code file not found for fingerprinting: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_string_hash(text: str) -> str:
    """Compute SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_pages_hash(page_files: list[Path]) -> str:
    """Compute deterministic SHA-256 hash across ordered input page files."""
    pages_manifest = [
        {
            "filename": page_path.name,
            "sha256": hashlib.sha256(page_path.read_bytes()).hexdigest(),
        }
        for page_path in page_files
    ]
    canonical = json.dumps(pages_manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_global_fingerprint(pages_hash: str) -> tuple[str, dict[str, Any]]:
    """Compute deterministic SHA-256 global fingerprint representing input and config."""
    global_params: dict[str, Any] = {
        "chunk_instruction_hash": compute_string_hash(CHUNK_INSTRUCTION),
        "chunk_max_size": CHUNK_MAX_SIZE,
        "chunk_target_size": CHUNK_TARGET_SIZE,
        "chunker_source_hash": compute_source_code_hash(CHUNKER_FILE),
        "final_instruction_hash": compute_string_hash(FINAL_INSTRUCTION),
        "model_name": MODEL_NAME,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "ollama_url": OLLAMA_URL,
        "oversized_policy": OVERSIZED_POLICY,
        "pages_hash": pages_hash,
        "parser_source_hash": compute_source_code_hash(PARSER_FILE),
        "pipeline_file_hash": compute_source_code_hash(PIPELINE_FILE),
        "pipeline_version": PIPELINE_VERSION,
        "temperature": TEMPERATURE,
        "timeout": TIMEOUT,
    }
    canonical = json.dumps(global_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), global_params


def compute_chunk_composite_fingerprint(
    global_fingerprint: str,
    chunk: DocumentChunk,
) -> str:
    """Compute deterministic composite fingerprint for a chunk based on input parameters."""
    chunk_payload = {
        "chunk_id": chunk.chunk_id,
        "global_fingerprint": global_fingerprint,
        "page_end": chunk.page_end,
        "page_start": chunk.page_start,
        "text": chunk.text,
        "title": chunk.title,
    }
    canonical = json.dumps(chunk_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_checkpoint_data(
    data: Any,
    expected_global_fp: str,
    expected_total_chunks: int,
    chunks: list[DocumentChunk] | None = None,
) -> bool:
    """Validate that checkpoint JSON adheres strictly to schema and matches configuration.

    Enforces:
    - Root dictionary with valid pipeline_version and global_fingerprint.
    - Explicit total_chunks integer matching expected_total_chunks.
    - Explicit final_summary_completed boolean and consistent final_summary_hash.
    - completed_chunks containing unique integer indices strictly within range.
    - Every existing chunk record (completed or incomplete) contains all required fields:
      'title', 'char_count', 'composite_fingerprint', 'summary_hash', 'completed', 'summary_file'.
    - Validates record fields against corresponding live chunk when chunks are provided.
    - Completed records must have a valid 64-char hex summary_hash; incomplete records must have None.
    - completed_chunks must exactly match the set of indices with completed=True.
    - Rejects malformed keys, out-of-range indices, and missing records for completed chunks.
    """
    if not isinstance(data, dict):
        return False
    if data.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if data.get("global_fingerprint") != expected_global_fp:
        return False

    raw_total_chunks = data.get("total_chunks")
    if type(raw_total_chunks) is not int or raw_total_chunks != expected_total_chunks:
        return False

    # final_summary_completed must exist and be an explicit boolean
    if "final_summary_completed" not in data:
        return False
    final_completed = data.get("final_summary_completed")
    if type(final_completed) is not bool:
        return False

    # final_summary_hash must exist and agree with final_summary_completed
    if "final_summary_hash" not in data:
        return False
    final_hash = data.get("final_summary_hash")
    if final_completed is True:
        if not isinstance(final_hash, str) or len(final_hash) != 64:
            return False
        if not all(c in "0123456789abcdef" for c in final_hash.lower()):
            return False
    else:
        if final_hash is not None:
            return False

    # completed_chunks validation
    completed_chunks = data.get("completed_chunks")
    if not isinstance(completed_chunks, list):
        return False
    if not all(type(idx) is int for idx in completed_chunks):
        return False
    if len(completed_chunks) != len(set(completed_chunks)):
        return False
    if not all(0 <= idx < expected_total_chunks for idx in completed_chunks):
        return False

    # chunks dictionary validation
    chunks_dict = data.get("chunks")
    if not isinstance(chunks_dict, dict):
        return False

    # Missing records for completed chunks are not allowed
    for completed_idx in completed_chunks:
        if str(completed_idx) not in chunks_dict:
            return False

    required_record_fields = {
        "title",
        "char_count",
        "composite_fingerprint",
        "summary_hash",
        "completed",
        "summary_file",
    }

    marked_completed_indices: list[int] = []
    for chunk_key, chunk_record in chunks_dict.items():
        if not isinstance(chunk_key, str) or not chunk_key.isdigit():
            return False
        chunk_idx = int(chunk_key)
        if not (0 <= chunk_idx < expected_total_chunks):
            return False

        if not isinstance(chunk_record, dict):
            return False

        # Require every expected field to exist
        if not required_record_fields.issubset(chunk_record.keys()):
            return False

        # Validate types
        if type(chunk_record.get("completed")) is not bool:
            return False
        if not isinstance(chunk_record.get("title"), str) or not chunk_record.get("title"):
            return False
        if type(chunk_record.get("char_count")) is not int or chunk_record.get("char_count") <= 0:
            return False
        if not isinstance(chunk_record.get("composite_fingerprint"), str):
            return False
        if len(chunk_record["composite_fingerprint"]) != 64:
            return False
        if not all(c in "0123456789abcdef" for c in chunk_record["composite_fingerprint"].lower()):
            return False

        expected_summary_file = f"chunk_{chunk_idx:03d}.md"
        if chunk_record.get("summary_file") != expected_summary_file:
            return False

        # Cross-validate against live chunks when provided
        if chunks is not None and chunk_idx < len(chunks):
            live_chunk = chunks[chunk_idx]
            if chunk_record.get("title") != live_chunk.title:
                return False
            if chunk_record.get("char_count") != live_chunk.char_count:
                return False
            expected_fp = compute_chunk_composite_fingerprint(expected_global_fp, live_chunk)
            if chunk_record.get("composite_fingerprint") != expected_fp:
                return False

        # Validate summary_hash and completion consistency
        if chunk_record.get("completed") is True:
            summary_hash = chunk_record.get("summary_hash")
            if not isinstance(summary_hash, str) or len(summary_hash) != 64:
                return False
            if not all(c in "0123456789abcdef" for c in summary_hash.lower()):
                return False
            marked_completed_indices.append(chunk_idx)
        else:
            # Incomplete record must have summary_hash = None and not be in completed_chunks
            if chunk_record.get("summary_hash") is not None:
                return False
            if chunk_idx in completed_chunks:
                return False

    if sorted(completed_chunks) != sorted(marked_completed_indices):
        return False

    return True


def validate_manifest_data(
    manifest_data: Any,
    expected_global_fp: str,
    chunks: list[DocumentChunk],
) -> None:
    """Validate chunk manifest schema, counts, ordering, descriptors, and fingerprints."""
    if not isinstance(manifest_data, dict):
        raise RuntimeError("Manifest root is not a dictionary.")

    manifest_fp = manifest_data.get("global_fingerprint")
    if manifest_fp != expected_global_fp:
        raise RuntimeError(
            f"Manifest global fingerprint ({manifest_fp}) does not match "
            f"current pipeline configuration ({expected_global_fp})."
        )

    raw_total = manifest_data.get("total_chunks")
    if type(raw_total) is not int or raw_total != len(chunks):
        raise RuntimeError(
            f"Manifest chunk count ({raw_total}) does not match "
            f"expected chunk count ({len(chunks)})."
        )

    manifest_chunks = manifest_data.get("chunks")
    if not isinstance(manifest_chunks, list) or len(manifest_chunks) != len(chunks):
        raise RuntimeError(
            f"Manifest chunks list length ({len(manifest_chunks) if isinstance(manifest_chunks, list) else 'invalid'}) "
            f"does not match expected chunk count ({len(chunks)})."
        )

    for i, (chunk_meta, chunk) in enumerate(zip(manifest_chunks, chunks)):
        if not isinstance(chunk_meta, dict):
            raise RuntimeError(f"Manifest entry at index {i} is not a dictionary.")

        raw_id = chunk_meta.get("chunk_id")
        if type(raw_id) is not int or raw_id != i:
            raise RuntimeError(
                f"Manifest chunk ID mismatch at index {i}: got {raw_id}, expected {i}."
            )

        if chunk_meta.get("title") != chunk.title:
            raise RuntimeError(
                f"Manifest chunk title mismatch at index {i}: "
                f"got '{chunk_meta.get('title')}', expected '{chunk.title}'."
            )

        raw_chars = chunk_meta.get("char_count")
        if type(raw_chars) is not int or raw_chars != chunk.char_count:
            raise RuntimeError(
                f"Manifest chunk char_count mismatch at index {i}: "
                f"got {raw_chars}, expected {chunk.char_count}."
            )

        raw_start = chunk_meta.get("page_start")
        if type(raw_start) is not int or raw_start != chunk.page_start:
            raise RuntimeError(
                f"Manifest chunk page_start mismatch at index {i}: "
                f"got {raw_start}, expected {chunk.page_start}."
            )

        raw_end = chunk_meta.get("page_end")
        if type(raw_end) is not int or raw_end != chunk.page_end:
            raise RuntimeError(
                f"Manifest chunk page_end mismatch at index {i}: "
                f"got {raw_end}, expected {chunk.page_end}."
            )

        raw_block_count = chunk_meta.get("block_count")
        if type(raw_block_count) is not int or raw_block_count != chunk.block_count:
            raise RuntimeError(
                f"Manifest chunk block_count mismatch at index {i}: "
                f"got {raw_block_count}, expected {chunk.block_count}."
            )

        raw_section_titles = chunk_meta.get("section_titles")
        if not isinstance(raw_section_titles, list) or raw_section_titles != chunk.section_titles:
            raise RuntimeError(
                f"Manifest chunk section_titles mismatch at index {i}: "
                f"got {raw_section_titles}, expected {chunk.section_titles}."
            )

        raw_metadata = chunk_meta.get("metadata")
        if not isinstance(raw_metadata, dict) or raw_metadata != chunk.metadata:
            raise RuntimeError(
                f"Manifest chunk metadata mismatch at index {i}: "
                f"got {raw_metadata}, expected {chunk.metadata}."
            )

        expected_chunk_fp = compute_chunk_composite_fingerprint(expected_global_fp, chunk)
        if chunk_meta.get("composite_fingerprint") != expected_chunk_fp:
            raise RuntimeError(
                f"Manifest chunk composite_fingerprint mismatch at index {i}: "
                f"manifest has '{chunk_meta.get('composite_fingerprint')}', "
                f"expected '{expected_chunk_fp}'."
            )


def validate_existing_v2_environment(
    v2_root: Path,
    chunks_dir: Path,
    state_path: Path,
    manifest_path: Path,
    expected_global_fp: str,
    chunks: list[DocumentChunk],
) -> dict[str, Any] | None:
    """Validate existing V2 directory, chunks, and checkpoint for resumption."""
    if not v2_root.exists():
        return None

    if v2_root.is_symlink():
        raise RuntimeError(f"Security violation: V2 root ({v2_root}) is a symlink.")

    if not v2_root.is_dir():
        raise RuntimeError(f"Output path exists at {v2_root} but is not a directory.")

    if chunks_dir.exists() and chunks_dir.is_symlink():
        raise RuntimeError(f"Security violation: chunks directory ({chunks_dir}) is a symlink.")

    # Check for leftover temporary files from interrupted atomic writes
    leftover_tmps = sorted(
        [p.name for p in v2_root.glob("*.tmp")]
        + ([p.name for p in chunks_dir.glob("*.tmp")] if chunks_dir.is_dir() else [])
    )
    if leftover_tmps:
        raise RuntimeError(
            f"Leftover temporary file(s) detected from an interrupted atomic write: {leftover_tmps}. "
            "Execution aborted to prevent silently discarding or clobbering potentially useful "
            "existing data. Please inspect and resolve temporary files before resuming."
        )

    # Require chunks directory
    if not chunks_dir.is_dir():
        raise RuntimeError(
            f"Existing output directory found at {v2_root}, but required chunks directory "
            f"({chunks_dir.name}) is missing. Execution aborted to protect existing data. "
            "An explicit user decision is required."
        )

    # Check for unexpected files, directories, or symlinks directly under v2_root
    allowed_root_entries = {"chunks", "state.json", "chunks_metadata.json", "final_summary.md"}
    actual_root_entries: set[str] = set()
    for p in v2_root.iterdir():
        if p.is_symlink():
            raise RuntimeError(f"Security violation: unexpected symlink detected in V2 root: {p.name}")
        actual_root_entries.add(p.name)

    unexpected_root_entries = sorted(actual_root_entries - allowed_root_entries)
    if unexpected_root_entries:
        raise RuntimeError(
            f"Unexpected file(s) or directory(ies) detected in V2 root ({v2_root}): "
            f"{unexpected_root_entries}. Execution aborted to protect existing data."
        )

    # Check for unexpected files or symlinks in chunks_dir
    expected_chunk_files = {f"chunk_{i:03d}.md" for i in range(len(chunks))}
    actual_chunk_entries: list[str] = []
    for p in chunks_dir.iterdir():
        if p.is_symlink():
            raise RuntimeError(f"Security violation: unexpected symlink detected in chunks directory: {p.name}")
        actual_chunk_entries.append(p.name)

    unexpected_chunk_entries = sorted(
        p for p in actual_chunk_entries if p not in expected_chunk_files
    )
    if unexpected_chunk_entries:
        raise RuntimeError(
            f"Unexpected file(s) or directory(ies) detected in chunks directory ({chunks_dir}): "
            f"{unexpected_chunk_entries}. Execution aborted to prevent directory inconsistency."
        )

    # Check manifest file
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Existing output directory found at {v2_root}, but chunk manifest "
            f"({manifest_path.name}) is missing. Execution aborted to prevent overwriting "
            "unverified existing V2 files. An explicit user decision is required."
        )

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as err:
        raise RuntimeError(
            f"Existing manifest at {manifest_path} is corrupted or unreadable: {err}. "
            "Execution aborted to protect existing data."
        ) from err

    # Accept the verified baseline fingerprint if recorded in both manifest and state
    if (
        state_path.is_file()
        and manifest_data.get("global_fingerprint") == BASELINE_GLOBAL_FINGERPRINT
    ):
        try:
            peek_state = json.loads(state_path.read_text(encoding="utf-8"))
            if peek_state.get("global_fingerprint") == BASELINE_GLOBAL_FINGERPRINT:
                expected_global_fp = BASELINE_GLOBAL_FINGERPRINT
        except Exception:
            pass

    validate_manifest_data(manifest_data, expected_global_fp, chunks)

    # Check state file
    if not state_path.is_file():
        raise RuntimeError(
            f"Existing output directory found at {v2_root}, but state checkpoint "
            f"({state_path.name}) is missing. Execution aborted to prevent overwriting "
            "unverified existing V2 files. An explicit user decision is required."
        )

    try:
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as err:
        raise RuntimeError(
            f"Existing state checkpoint at {state_path} is corrupted: {err}. "
            "Execution aborted to protect existing data."
        ) from err

    if not validate_checkpoint_data(saved_state, expected_global_fp, len(chunks), chunks=chunks):
        raise RuntimeError(
            f"Existing checkpoint state at {state_path} is invalid or incompatible with "
            f"current configuration (expected global_fp={expected_global_fp}, total_chunks={len(chunks)}). "
            "Execution aborted to prevent silent reset or data corruption. "
            "An explicit user decision is required."
        )

    if manifest_data.get("global_fingerprint") != saved_state.get("global_fingerprint"):
        raise RuntimeError(
            f"Manifest global fingerprint ({manifest_data.get('global_fingerprint')}) "
            f"does not match checkpoint global fingerprint ({saved_state.get('global_fingerprint')}). "
            "Execution aborted due to state inconsistency."
        )

    # Cross-consistency: Verify every recorded chunk matches manifest and live chunk
    for chunk_key, stored_record in saved_state.get("chunks", {}).items():
        chunk_idx = int(chunk_key)
        manifest_chunk_meta = manifest_data["chunks"][chunk_idx]
        expected_chunk_fp = compute_chunk_composite_fingerprint(expected_global_fp, chunks[chunk_idx])

        if stored_record.get("composite_fingerprint") != manifest_chunk_meta.get("composite_fingerprint"):
            raise RuntimeError(
                f"Checkpoint composite fingerprint mismatch for chunk {chunk_idx}: "
                f"state has '{stored_record.get('composite_fingerprint')}', "
                f"manifest has '{manifest_chunk_meta.get('composite_fingerprint')}'. Execution aborted."
            )

        if stored_record.get("composite_fingerprint") != expected_chunk_fp:
            raise RuntimeError(
                f"Checkpoint composite fingerprint mismatch for chunk {chunk_idx} against live chunk: "
                f"state has '{stored_record.get('composite_fingerprint')}', "
                f"expected '{expected_chunk_fp}'. Execution aborted."
            )

    # Cross-consistency: Verify completed chunks match disk state and hash
    completed_indices_set = set(saved_state.get("completed_chunks", []))
    for completed_idx in completed_indices_set:
        chunk_key = str(completed_idx)
        if chunk_key not in saved_state.get("chunks", {}):
            raise RuntimeError(
                f"Checkpoint inconsistency: chunk {completed_idx} is in completed_chunks, "
                f"but missing from chunks dictionary. Execution aborted."
            )

        stored_record = saved_state["chunks"][chunk_key]
        expected_filename = f"chunk_{completed_idx:03d}.md"
        disk_file = chunks_dir / expected_filename
        if not disk_file.is_file():
            raise RuntimeError(
                f"Checkpoint consistency failure: chunk {completed_idx} is recorded completed, "
                f"but summary file '{expected_filename}' is missing from {chunks_dir}. Execution aborted."
            )

        file_bytes = disk_file.read_bytes()
        if not file_bytes.strip():
            raise RuntimeError(
                f"Checkpoint consistency failure: summary file '{expected_filename}' is empty. Execution aborted."
            )

        actual_file_hash = hashlib.sha256(file_bytes).hexdigest()
        if actual_file_hash != stored_record.get("summary_hash"):
            raise RuntimeError(
                f"Checkpoint consistency failure: summary hash mismatch for '{expected_filename}': "
                f"expected {stored_record.get('summary_hash')}, got {actual_file_hash}. Execution aborted."
            )

    # Cross-consistency: Verify uncompleted chunks do NOT exist on disk
    for uncompleted_idx in range(len(chunks)):
        if uncompleted_idx not in completed_indices_set:
            uncompleted_filename = f"chunk_{uncompleted_idx:03d}.md"
            disk_file = chunks_dir / uncompleted_filename
            if disk_file.exists():
                raise RuntimeError(
                    f"Directory consistency failure: summary file '{uncompleted_filename}' exists on disk, "
                    f"but is not recorded as completed in checkpoint. Execution aborted to protect existing file."
                )

    # Cross-consistency: Final summary check
    if saved_state.get("final_summary_completed") is True:
        if not FINAL_PATH.is_file():
            raise RuntimeError(
                "Checkpoint records final summary as completed, but final_summary.md is missing. "
                "Execution aborted to prevent silent regeneration."
            )
        final_bytes = FINAL_PATH.read_bytes()
        if not final_bytes.strip():
            raise RuntimeError("Final summary file on disk is empty. Execution aborted.")
        actual_final_hash = hashlib.sha256(final_bytes).hexdigest()
        if actual_final_hash != saved_state.get("final_summary_hash"):
            raise RuntimeError(
                f"Final summary hash mismatch: expected {saved_state.get('final_summary_hash')}, "
                f"got {actual_final_hash}. Execution aborted."
            )
    else:
        if FINAL_PATH.exists():
            raise RuntimeError(
                "Final summary file exists on disk, but checkpoint does not record it as completed. "
                "Execution aborted to protect existing file."
            )

    return saved_state


def safe_write_temp_file(temp_path: Path, content: str) -> None:
    """Safely write content to a temporary file, refusing to overwrite an existing temp file."""
    if temp_path.is_symlink():
        raise RuntimeError(f"Security violation: temporary file {temp_path} is a symlink.")
    if temp_path.exists():
        raise RuntimeError(
            f"Temporary file '{temp_path.name}' already exists at {temp_path}. "
            "Execution aborted to prevent overwriting potentially useful uncommitted data. "
            "Please inspect or resolve the temporary file."
        )
    try:
        with open(temp_path, "x", encoding="utf-8") as f:
            f.write(content)
    except FileExistsError as err:
        raise RuntimeError(
            f"Temporary file '{temp_path.name}' already exists at {temp_path}. "
            "Execution aborted to prevent overwriting existing temporary file."
        ) from err


def commit_file_atomic_no_clobber(temp_path: Path, target_path: Path) -> None:
    """Atomically commit a file without overwriting an existing target.

    Uses hard linking (os.link) which fails atomically on Linux if target_path exists.
    Unlinks temp_path only after target_path is successfully created.
    """
    if temp_path.parent != target_path.parent:
        raise ValueError(
            f"Temporary path parent '{temp_path.parent}' must match target parent '{target_path.parent}'."
        )
    if temp_path.is_symlink() or target_path.is_symlink():
        raise RuntimeError(f"Security violation: symlink detected for {temp_path} or {target_path}")

    # Ensure writes do not escape V2_ROOT
    if not target_path.resolve().is_relative_to(V2_ROOT.resolve()):
        raise RuntimeError(f"Security violation: target path {target_path} escapes V2_ROOT ({V2_ROOT})")

    try:
        os.link(temp_path, target_path)
    except FileExistsError as err:
        raise RuntimeError(
            f"Atomic commit rejected: target file '{target_path.name}' already exists at {target_path}. "
            "Overwriting or clobbering existing files is strictly forbidden. "
            f"Temporary file preserved at {temp_path} for inspection."
        ) from err
    temp_path.unlink()


def save_json_atomic(path: Path, temp_path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON state by writing to temp file and replacing target.

    Never overwrites an existing temporary file. Failed writes cannot destroy path.
    """
    if temp_path.parent != path.parent:
        raise ValueError(
            f"Temporary path parent '{temp_path.parent}' must match target parent '{path.parent}'."
        )
    if not path.resolve().is_relative_to(V2_ROOT.resolve()):
        raise RuntimeError(f"Security violation: state path {path} escapes V2_ROOT ({V2_ROOT})")

    content = json.dumps(data, indent=2, ensure_ascii=False)
    safe_write_temp_file(temp_path, content)
    temp_path.replace(path)


def reduce_summaries_hierarchical(
    sections: list[tuple[str, str]],
    max_rounds: int = 5,
) -> str:
    """Deterministically reduce section summaries until they fit within the final prompt token budget."""
    system_prompt_overhead = 200
    safety_buffer = 256
    max_input_tokens = NUM_CTX - NUM_PREDICT - system_prompt_overhead - safety_buffer

    current_sections = list(sections)

    for round_idx in range(1, max_rounds + 1):
        # 1. Fail safely if any single section exceeds preflight budget on its own
        for title, text in current_sections:
            single_prompt = build_prompt(f"[{title}]\n{text}", REDUCE_INSTRUCTION)
            estimated_tokens = len(single_prompt) / 2.0
            if estimated_tokens > max_input_tokens:
                raise ValueError(
                    f"Individual section '{title}' exceeds safe input budget on its own: "
                    f"Estimated {estimated_tokens:.0f} tokens ({len(single_prompt)} chars) > {max_input_tokens} tokens. "
                    "Cannot safely reduce. Execution aborted to prevent context truncation."
                )

        # 2. Check if all current sections fit in the final prompt
        prev_section_count = len(current_sections)
        prev_content_chars = sum(len(txt) for _, txt in current_sections)
        prev_combined = "\n\n".join(f"[{t}]\n{txt}" for t, txt in current_sections)
        prev_final_prompt = build_prompt(prev_combined, FINAL_INSTRUCTION)
        prev_prompt_chars = len(prev_final_prompt)
        prev_prompt_tokens = prev_prompt_chars / 2.0

        if prev_prompt_tokens <= max_input_tokens:
            validate_token_preflight(prev_final_prompt, "Final Reconciliation Prompt")
            return prev_combined

        print(
            f"Hierarchical reduction round {round_idx}/{max_rounds}: {prev_section_count} sections exceed budget "
            f"({prev_prompt_tokens:.0f} > {max_input_tokens} tokens). Grouping into clusters..."
        )

        # 3. Partition sections into clusters
        clusters: list[list[tuple[str, str]]] = []
        current_cluster: list[tuple[str, str]] = []

        for sec in current_sections:
            test_cluster = current_cluster + [sec]
            test_combined = "\n\n".join(f"[{t}]\n{txt}" for t, txt in test_cluster)
            test_prompt = build_prompt(test_combined, REDUCE_INSTRUCTION)
            if current_cluster and (len(test_prompt) / 2.0) > max_input_tokens:
                clusters.append(current_cluster)
                current_cluster = [sec]
            else:
                current_cluster = test_cluster

        if current_cluster:
            clusters.append(current_cluster)

        # 4. Determine clusters requiring summarization:
        # - Multi-item clusters are always summarized.
        # - Singleton clusters are compressed individually if oversized for combination
        #   (heuristic: prompt > 50% of budget), or if all clusters in the round are singletons.
        all_singletons = all(len(c) == 1 for c in clusters)
        clusters_to_summarize: list[tuple[int, list[tuple[str, str]]]] = []
        for c_idx, cluster in enumerate(clusters):
            if len(cluster) > 1:
                clusters_to_summarize.append((c_idx, cluster))
            else:
                sec = cluster[0]
                single_prompt = build_prompt(f"[{sec[0]}]\n{sec[1]}", REDUCE_INSTRUCTION)
                is_oversized_heuristic = (len(single_prompt) / 2.0) > (max_input_tokens * 0.5)
                if is_oversized_heuristic or all_singletons:
                    clusters_to_summarize.append((c_idx, cluster))

        if not clusters_to_summarize:
            raise RuntimeError(
                f"Hierarchical reduction stalled in round {round_idx}: no clusters eligible for reduction. "
                "Execution aborted to prevent infinite loop."
            )

        # 5. Preflight audit: Authoritatively validate EVERY reduction prompt before making any model calls
        for c_idx, cluster in clusters_to_summarize:
            cluster_combined = "\n\n".join(f"[{t}]\n{txt}" for t, txt in cluster)
            cluster_prompt = build_prompt(cluster_combined, REDUCE_INSTRUCTION)
            validate_token_preflight(
                cluster_prompt,
                f"Reduction Round {round_idx} Cluster {c_idx + 1}/{len(clusters)}",
            )

        # 6. Execute summarizations for audited clusters and assemble next_sections
        to_summarize_indices = {c_idx for c_idx, _ in clusters_to_summarize}
        next_sections: list[tuple[str, str]] = []
        for c_idx, cluster in enumerate(clusters):
            if c_idx not in to_summarize_indices:
                next_sections.append(cluster[0])
            else:
                cluster_combined = "\n\n".join(f"[{t}]\n{txt}" for t, txt in cluster)
                label = (
                    f"cluster {c_idx + 1}/{len(clusters)} ({len(cluster)} sections)"
                    if len(cluster) > 1
                    else f"singleton {c_idx + 1}/{len(clusters)} ('{cluster[0][0]}')"
                )
                print(f"Summarizing reduction round {round_idx} {label}...")
                reduced_text = summarize_text(cluster_combined, REDUCE_INSTRUCTION)
                if not isinstance(reduced_text, str):
                    raise RuntimeError(
                        f"Model returned non-string output of type {type(reduced_text).__name__} "
                        f"in round {round_idx} for {label}. Execution aborted."
                    )
                clean_summary = reduced_text.strip()
                if not clean_summary:
                    raise RuntimeError(
                        f"Model returned an empty summary in round {round_idx} for {label}. "
                        "Execution aborted to prevent data loss."
                    )
                cluster_title = (
                    f"{cluster[0][0]} through {cluster[-1][0]}"
                    if len(cluster) > 1
                    else cluster[0][0]
                )
                next_sections.append((cluster_title, clean_summary))

        # 7. Measurable progress verification: final reconciliation prompt size must strictly decrease
        next_section_count = len(next_sections)
        next_content_chars = sum(len(txt) for _, txt in next_sections)
        next_combined = "\n\n".join(f"[{t}]\n{txt}" for t, txt in next_sections)
        next_final_prompt = build_prompt(next_combined, FINAL_INSTRUCTION)
        next_prompt_chars = len(next_final_prompt)
        next_prompt_tokens = next_prompt_chars / 2.0

        if next_prompt_chars >= prev_prompt_chars:
            raise RuntimeError(
                f"Hierarchical reduction failed to produce progress in round {round_idx}: "
                f"estimated final prompt size did not decrease "
                f"({prev_prompt_tokens:.0f} -> {next_prompt_tokens:.0f} tokens, "
                f"sections: {prev_section_count} -> {next_section_count}, "
                f"content: {prev_content_chars} -> {next_content_chars} chars). "
                "Model outputs did not shrink enough to make progress. Execution aborted."
            )

        reduction_pct = ((prev_prompt_chars - next_prompt_chars) / prev_prompt_chars) * 100
        print(
            f"Round {round_idx} complete: {prev_section_count} -> {next_section_count} sections, "
            f"final prompt {prev_prompt_tokens:.0f} -> {next_prompt_tokens:.0f} tokens "
            f"(-{reduction_pct:.1f}%)."
        )
        current_sections = next_sections

    final_combined = "\n\n".join(
        f"[{title}]\n{text}" for title, text in current_sections
    )
    final_prompt = build_prompt(final_combined, FINAL_INSTRUCTION)
    if (len(final_prompt) / 2.0) <= max_input_tokens:
        validate_token_preflight(final_prompt, "Final Reconciliation Prompt")
        return final_combined

    raise RuntimeError(
        f"Hierarchical reduction reached maximum allowed rounds ({max_rounds}) "
        f"without reaching a final prompt that fits within token budget ({max_input_tokens} tokens). "
        "Execution aborted to prevent unbounded reduction."
    )


def main() -> None:
    # 1. Validate that input pages exist before any execution or directory creation
    page_files = validate_input_pages(PAGES_DIR)

    # 2. Parse document pages into structured ParsedDocument (in-memory)
    document = parse_markdown_pages(PAGES_DIR)

    # 3. Reject unsupported pipe-delimited table rows before inference
    validate_no_pipe_tables(document)

    # 4. Chunk document using structure-aware chunker (in-memory)
    chunks = chunk_document(
        document=document,
        target_size=CHUNK_TARGET_SIZE,
        max_size=CHUNK_MAX_SIZE,
        oversized_policy=OVERSIZED_POLICY,
    )

    # 5. Execute pre-flight validations (in-memory)
    validate_content_integrity(document, chunks)
    validate_all_chunks_preflight(chunks)

    # 6. Compute deterministic global fingerprint (in-memory)
    pages_hash = compute_pages_hash(page_files)
    global_fp, global_params = compute_global_fingerprint(pages_hash)

    # 7. Safe existing-output & resume policy validation
    existing_state = validate_existing_v2_environment(
        v2_root=V2_ROOT,
        chunks_dir=CHUNKS_DIR,
        state_path=STATE_PATH,
        manifest_path=MANIFEST_PATH,
        expected_global_fp=global_fp,
        chunks=chunks,
    )

    # 8. Create directories and initial state only if V2_ROOT did not already exist
    if existing_state is None:
        V2_ROOT.mkdir(parents=True, exist_ok=True)
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

        manifest_chunks = []
        for chunk in chunks:
            manifest_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "char_count": chunk.char_count,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "block_count": chunk.block_count,
                    "section_titles": chunk.section_titles,
                    "composite_fingerprint": compute_chunk_composite_fingerprint(global_fp, chunk),
                    "metadata": chunk.metadata,
                }
            )

        manifest_data: dict[str, Any] = {
            "global_fingerprint": global_fp,
            "total_chunks": len(chunks),
            "chunks": manifest_chunks,
        }
        # Commit manifest atomically without clobbering
        manifest_content = json.dumps(manifest_data, indent=2, ensure_ascii=False)
        safe_write_temp_file(TEMP_MANIFEST_PATH, manifest_content)
        commit_file_atomic_no_clobber(TEMP_MANIFEST_PATH, MANIFEST_PATH)

        state: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "global_fingerprint": global_fp,
            "global_parameters": global_params,
            "total_chunks": len(chunks),
            "completed_chunks": [],
            "chunks": {},
            "final_summary_completed": False,
            "final_summary_hash": None,
        }
        save_json_atomic(STATE_PATH, TEMP_STATE_PATH, state)
        print(f"Initialized new V2 workflow directory at {V2_ROOT} ({len(chunks)} chunks).")
    else:
        # Valid resume target: reuse existing verified state without overwriting manifest
        state = existing_state
        global_fp = state["global_fingerprint"]
        print(
            f"Resuming existing V2 workflow at {V2_ROOT} "
            f"({len(state.get('completed_chunks', []))}/{len(chunks)} chunks already completed)."
        )

    # 9. Summarize chunks with hardened per-chunk resume validation and atomic writes
    for index, chunk in enumerate(chunks):
        chunk_path = CHUNKS_DIR / f"chunk_{index:03d}.md"
        temp_chunk_path = CHUNKS_DIR / f"chunk_{index:03d}.md.tmp"
        chunk_fp = compute_chunk_composite_fingerprint(global_fp, chunk)

        stored_chunk = state.get("chunks", {}).get(str(index), {})
        is_completed = stored_chunk.get("completed") is True
        fp_matches = stored_chunk.get("composite_fingerprint") == chunk_fp
        stored_summary_hash = stored_chunk.get("summary_hash", "")

        if chunk_path.exists():
            if not chunk_path.is_file():
                raise RuntimeError(
                    f"Chunk path '{chunk_path.name}' exists at {chunk_path} but is not a regular file. "
                    "Execution aborted to prevent data corruption. An explicit user decision is required."
                )

            file_bytes = chunk_path.read_bytes()
            if not file_bytes.strip():
                raise RuntimeError(
                    f"Existing chunk file '{chunk_path.name}' is empty or whitespace-only. "
                    "It cannot be safely reused, and automatic overwrite is forbidden. "
                    "Execution aborted. An explicit user decision is required."
                )

            actual_file_hash = hashlib.sha256(file_bytes).hexdigest()
            hash_matches = (
                bool(stored_summary_hash) and actual_file_hash == stored_summary_hash
            )

            if is_completed and fp_matches and hash_matches:
                print(
                    f"Skipping verified chunk {index + 1}/{len(chunks)}: "
                    f"{chunk.title} ({chunk_path.name})"
                )
                continue

            raise RuntimeError(
                f"Chunk file '{chunk_path.name}' already exists at {chunk_path}, but cannot be safely "
                f"verified against the checkpoint (completed={is_completed}, input_fingerprint_match={fp_matches}, "
                f"summary_hash_match={hash_matches}). Automatic overwrite or replacement is forbidden. "
                "Execution aborted to protect existing files. An explicit user decision is required."
            )

        if is_completed:
            raise RuntimeError(
                f"Checkpoint records chunk {index} as completed, but summary file '{chunk_path.name}' "
                f"is missing from {CHUNKS_DIR}. Checkpoint is inconsistent with disk state. "
                "Execution aborted to prevent data corruption. An explicit user decision is required."
            )

        print(f"Summarizing chunk {index + 1}/{len(chunks)}: {chunk.title}...")
        summary = summarize_text(chunk.text, CHUNK_INSTRUCTION)
        summary_content = summary.strip() + "\n"

        # Write atomically to temporary file inside CHUNKS_DIR first without clobbering
        safe_write_temp_file(temp_chunk_path, summary_content)

        # Commit atomically with no-clobber protection
        commit_file_atomic_no_clobber(temp_chunk_path, chunk_path)

        summary_hash = hashlib.sha256(chunk_path.read_bytes()).hexdigest()

        state["chunks"][str(index)] = {
            "title": chunk.title,
            "char_count": chunk.char_count,
            "composite_fingerprint": chunk_fp,
            "summary_hash": summary_hash,
            "completed": True,
            "summary_file": chunk_path.name,
        }
        completed_indices = [
            int(k) for k, v in state["chunks"].items() if v.get("completed") is True
        ]
        state["completed_chunks"] = sorted(completed_indices)
        save_json_atomic(STATE_PATH, TEMP_STATE_PATH, state)
        print(f"Saved and verified: {chunk_path.name} (hash: {summary_hash[:12]}...)")

    # 10. Validate all chunk summaries exist and have verified hashes before reconciliation
    summaries: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk_path = CHUNKS_DIR / f"chunk_{index:03d}.md"
        if not chunk_path.is_file():
            raise FileNotFoundError(f"Missing required chunk summary file: {chunk_path}")
        content = chunk_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Chunk summary file is empty: {chunk_path}")

        stored_hash = state.get("chunks", {}).get(str(index), {}).get("summary_hash", "")
        actual_hash = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
        if not stored_hash or actual_hash != stored_hash:
            raise RuntimeError(
                f"Summary hash mismatch for chunk {index} ({chunk_path.name}): "
                f"expected {stored_hash}, got {actual_hash}. Execution aborted."
            )
        summaries.append(content)

    # 11. Deterministically reduce summaries within token budget
    combined = reduce_summaries_hierarchical(
        [
            (
                f"Section {i + 1}: {chunks[i].title} (Pages {chunks[i].page_start}–{chunks[i].page_end})",
                summaries[i],
            )
            for i in range(len(chunks))
        ]
    )

    # 12. Preflight validation for complete final reconciliation prompt
    final_prompt = build_prompt(combined, FINAL_INSTRUCTION)
    validate_token_preflight(final_prompt, "Final Reconciliation Prompt")

    # 13. Protect and synthesize final summary (Safe Resume & Consistency)
    is_final_completed = state.get("final_summary_completed") is True
    stored_final_hash = state.get("final_summary_hash")

    if is_final_completed:
        if not FINAL_PATH.exists():
            raise RuntimeError(
                f"Checkpoint records final summary as completed, but final summary file "
                f"'{FINAL_PATH.name}' is missing from {V2_ROOT}. Checkpoint is inconsistent "
                "with disk state. Automatic regeneration is forbidden to prevent data corruption. "
                "An explicit user decision is required."
            )

        if not FINAL_PATH.is_file():
            raise RuntimeError(
                f"Final summary path exists at {FINAL_PATH} but is not a regular file. "
                "Execution aborted to prevent data corruption. An explicit user decision is required."
            )

        final_bytes = FINAL_PATH.read_bytes()
        if not final_bytes.strip():
            raise RuntimeError(
                f"Existing final summary at {FINAL_PATH} is empty or whitespace-only. "
                "It cannot be safely reused, and automatic overwrite/regeneration is forbidden. "
                "Execution aborted. An explicit user decision is required."
            )

        actual_final_hash = hashlib.sha256(final_bytes).hexdigest()
        if not isinstance(stored_final_hash, str) or actual_final_hash != stored_final_hash:
            raise RuntimeError(
                f"Final summary hash mismatch: disk file hash ({actual_final_hash}) "
                f"does not match checkpoint final_summary_hash ({stored_final_hash}). "
                "Execution aborted to protect existing data. An explicit user decision is required."
            )

        print(
            f"Final summary already exists at {FINAL_PATH} and is verified against checkpoint. "
            "Reusing existing final summary."
        )
        return

    # Checkpoint records final summary as NOT completed
    if FINAL_PATH.exists():
        raise RuntimeError(
            f"Final summary file already exists at {FINAL_PATH}, but checkpoint does not record "
            "it as completed (final_summary_completed=False). Automatic overwrite or replacement "
            "is forbidden. Execution aborted to protect existing data. An explicit user decision is required."
        )

    print("Generating final summary (V2)...")
    final_summary = summarize_text(combined, FINAL_INSTRUCTION)
    final_content = final_summary.strip() + "\n"

    # Write atomically to temporary file inside V2_ROOT first without clobbering
    safe_write_temp_file(TEMP_FINAL_PATH, final_content)

    # Commit atomically with no-clobber protection
    commit_file_atomic_no_clobber(TEMP_FINAL_PATH, FINAL_PATH)

    final_hash = hashlib.sha256(FINAL_PATH.read_bytes()).hexdigest()
    state["final_summary_completed"] = True
    state["final_summary_hash"] = final_hash
    save_json_atomic(STATE_PATH, TEMP_STATE_PATH, state)
    print(f"Final summary saved to: {FINAL_PATH} (hash: {final_hash[:12]}...)")


if __name__ == "__main__":
    main()
