# Project Context: PDF Audio Generator

## 1. Project Overview

The **PDF Audio Generator** is a Python-based processing pipeline designed to convert complex technical and academic PDF documents (such as research papers) into structured summaries and conversational, podcast-style audio narration.

The system is designed to run locally using open-source models:
- **OCR Engine**: `indra_ocr` (locally installed from `./OCR/indra`) for layout analysis and Markdown extraction from PDF pages.
- **LLM Engine**: Local Ollama server (`http://localhost:11434/api/generate`) running `qwen2.5:3b-instruct` with a default context window of `num_ctx = 4096`.
- **TTS Engine**: `gTTS` (Google Text-to-Speech) for converting generated scripts into MP3 audio.

The current reference document used for end-to-end benchmarking is the seminal paper *"Attention Is All You Need"* (Vaswani et al., 2017), spanning 15 pages.

---

## 2. Current Architecture and Data Flow

The project consists of six distinct functional stages:

```
[Source PDF]
     │
     │ Stage 1: OCR Extraction (backend/tests/test_large_pdf.py via indra_ocr)
     ▼
[outputs/large_pdf/pages/page_*.md] (15 OCR Markdown pages)
     │
     ├───► Stage 2: Document Parsing (backend/document_parser/parser.py)
     │          Produces ParsedDocument (tested in backend/tests/test_document_parser.py)
     │          [CURRENTLY DECOUPLED FROM STAGE 3]
     │
     └───► Stage 3: Chunking & Summarization (backend/tests/test_large_pdf_summary.py)
                Bypasses parser; concatenates raw page_*.md files directly.
                Slices raw text with naive character window (CHUNK_SIZE=6000, CHUNK_OVERLAP=500).
                Calls Ollama qwen2.5:3b-instruct chunk-by-chunk.
                │
                ▼
           [outputs/large_pdf/summary/chunks/chunk_*.md] (9 chunks)
                │
                ▼
           [outputs/large_pdf/summary/final_summary.md]
                │
                ├─► Stage 4: Audio Script Generation (backend/tests/test_large_pdf_audio_script.py)
                │        Calls Ollama to create natural conversational narration.
                │        Output: outputs/large_pdf/audio_script.md
                │
                └─► Stage 5: Text-to-Speech (backend/tests/test_tts.py via gTTS)
                         Converts script to MP3 audio.
                         Output: outputs/audio.mp3
```

---

## 3. Important Source Files

| Component / Stage | File Path | Description |
| :--- | :--- | :--- |
| **Document Parser** | `backend/document_parser/parser.py` | Core parser module. Implements `parse_markdown_pages(pages_dir)` returning `ParsedDocument`. |
| **Parser Package Init** | `backend/document_parser/__init__.py` | Exports `parse_markdown_pages`, `ParsedDocument`, `Section`, and `ContentBlock`. |
| **Parser Unit Tests** | `backend/tests/test_document_parser.py` | Unit tests for parser structural fidelity, section order, table preservation, and page boundary handling. |
| **Large PDF OCR Extraction** | `backend/tests/test_large_pdf.py` | Runs `indra_ocr` over input PDF, generating `page_0000.md` through `page_0014.md` and `outputs/large_pdf/state.json`. |
| **Chunking & Summarization** | `backend/tests/test_large_pdf_summary.py` | Current script for chunking raw Markdown pages and summarizing them with Ollama. |
| **Audio Script Generation** | `backend/tests/test_large_pdf_audio_script.py` | Generates a podcast-style audio script from `outputs/large_pdf/summary/final_summary.md`. |
| **Small PDF Summary Test** | `backend/tests/test_pdf_summary.py` | Earlier prototype summarizing the first 3 pages of a PDF. |
| **Small Script Test** | `backend/tests/test_audio_script.py` | Earlier prototype generating script from `outputs/summary.md`. |
| **TTS Generation** | `backend/tests/test_tts.py` | Converts `outputs/audio_script.md` to `outputs/audio.mp3` using `gTTS`. |
| **Model Verification Tests** | `backend/tests/test_qwen.py`, `backend/tests/test_indra.py` | Minimal connectivity tests for Ollama and Indra OCR. |

---

## 4. ParsedDocument / Section / ContentBlock Design

The parser in `backend/document_parser/parser.py` defines three primary dataclasses:

### 4.1 Data Structures
```python
@dataclass
class ContentBlock:
    type: str          # "paragraph", "heading", "html_table", "table_row"
    text: str          # Raw text or HTML markup of the block
    page: int          # 1-indexed PDF page number
    order: int = 0     # Monotonic global reading order across the document
    needs_review: bool = False

@dataclass
class Section:
    title: str
    level: int         # Heading level (e.g. 1 for ##, 2 for ###)
    page_start: int
    page_end: int
    content_blocks: list[ContentBlock] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)  # Currently flat; unused
    needs_review: bool = False

@dataclass
class ParsedDocument:
    source: str
    pages: int
    content_blocks: list[ContentBlock] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    unassigned_blocks: list[ContentBlock] = field(default_factory=list)
```

### 4.2 Element Representation Details
- **Headings**: Detected using regex (`HEADING_PATTERN`, `NUMBERED_SECTION_PATTERN`) and filtered by `is_likely_heading()` (which suppresses mathematical equations and long strings mistakenly formatted as headings). A valid heading creates a `Section` in `document.sections` and a `ContentBlock(type="heading")` in `document.content_blocks`.
- **Sections**: Kept in a flat list in `ParsedDocument.sections`. `current_section` accumulates subsequent blocks until the next section heading. Sections track `page_start` and `page_end`. Any blocks occurring before the first recognized heading (e.g. title, authors, affiliations) are stored in `document.unassigned_blocks`.
- **Paragraphs**: Accumulated lines flushed at blank lines, table boundaries, headings, or page boundaries. `flush_paragraph()` guarantees that paragraphs do not span across page boundaries, preserving accurate per-page attribution.
- **Tables**: Complete HTML tables (`<table>...</table>`) are parsed as single atomic `ContentBlock(type="html_table")`. Markdown tables with pipe syntax are parsed line-by-line as `ContentBlock(type="table_row")`. In the 15-page reference document, all 13 tables are HTML tables.
- **Equations**: Display equations wrapped in HTML tables are captured as `type="html_table"`. Equations mistakenly formatted with heading syntax (e.g. `## MultiHead(...)`) are caught by heuristic rejection and parsed as `type="paragraph"`. Inline math remains inside paragraph text. There is no dedicated `equation` block type.
- **Ordering**: Every block receives a strictly sequential 1-based `order` counter matching document visual flow.

---

## 5. Existing Outputs and Preservation Rules

The following generated artifacts exist on disk and represent significant compute or verified benchmark state:

| Path | Description | Preservation Status |
| :--- | :--- | :--- |
| `outputs/large_pdf/pages/page_*.md` | 15 extracted Markdown pages (`page_0000.md` to `page_0014.md`) | **DO NOT DELETE OR OVERWRITE** |
| `outputs/large_pdf/state.json` | OCR extraction state recording completed pages | **DO NOT DELETE OR OVERWRITE** |
| `outputs/large_pdf/parser/document_structure.json` | Serialized `ParsedDocument` output | **DO NOT DELETE OR OVERWRITE** |
| `outputs/large_pdf/summary/chunks/chunk_*.md` | 9 chunk summaries (`chunk_000.md` through `chunk_008.md`) | **DO NOT DELETE OR OVERWRITE** |
| `outputs/large_pdf/summary/state.json` | Chunk summarization state file tracking completed chunks | **DO NOT DELETE OR OVERWRITE** |
| `outputs/large_pdf/summary/final_summary.md` | Reconciled comprehensive summary (8,853 bytes) | **DO NOT DELETE OR OVERWRITE** |
| `outputs/large_pdf/audio_script.md` | Educational podcast-style script generated from summary | **DO NOT DELETE OR OVERWRITE** |
| `outputs/summary.md` | 3-page prototype summary | **DO NOT DELETE OR OVERWRITE** |
| `outputs/audio_script.md` | 3-page prototype audio script | **DO NOT DELETE OR OVERWRITE** |
| `outputs/audio.mp3` | Generated audio narration MP3 | **DO NOT DELETE OR OVERWRITE** |

### Strict Rule on Existing Outputs
Existing outputs must never be removed, truncated, or regenerated unless explicit user authorization is granted. Any testing of new components must occur in isolated temporary directories (`tempfile.TemporaryDirectory`) or use non-colliding paths.

---

## 6. Confirmed Findings

1. **Pipeline Decoupling**: The structured parser `backend/document_parser/parser.py` is tested in isolation (`backend/tests/test_document_parser.py`), but the summarization script (`backend/tests/test_large_pdf_summary.py`) completely bypasses it, reading raw `.md` files via `load_pages()`.
2. **Naive Slicing Defects**:
   - Fixed character slicing (`CHUNK_SIZE = 6000`, `CHUNK_OVERLAP = 500`) cuts across words, sentences, HTML table tags, and mathematical equations.
   - Example: Chunk 0 cuts mid-sentence on page 3, leaving orphaned HTML tags at the start of Chunk 1.
3. **Harmful Overlap**: Arbitrary 500-character overlap duplicates trailing and leading sentence fragments, confusing the 3B model and introducing redundancy in chunk summaries.
4. **Absence of Section Context**: The LLM receives chunks without section headers, obscuring the logical role of the text (e.g. background vs. methodology vs. results).
5. **Block Size Distribution**:
   - Total document blocks in reference paper: 133 (90 paragraphs, 30 headings, 13 HTML tables).
   - Max individual block size: ~3,204 characters (Table 3 in Section 6.2).
   - All blocks fit well within a 5,000-character target budget, meaning block-level chunking will rarely or never need to split individual blocks.

---

## 7. Architectural Decisions

1. **Structure-Aware Chunking**:
   - Replace character slicing with an algorithm that consumes `ParsedDocument`.
   - Use `Section` as the primary boundary.
   - Coalesce small consecutive sections into single chunks up to a target size of **4,500 to 5,500 characters** (~1,100–1,400 tokens) to safely fit Qwen 2.5 3B with `num_ctx = 4096`.
2. **Block Atomicity**:
   - `ContentBlock` instances (including `html_table` and equations) must be kept atomic. Chunks must never split an individual block unless a single block exceeds the maximum chunk size.
   - For oversized sections (> 5,500 chars), split along block boundaries and prepend continuation headers (e.g. `### Section Name (Part 2)`).
3. **Elimination of Character Overlap**:
   - Discard raw character sliding window overlap.
   - Replace with contextual breadcrumb headers (e.g. `[Context: Continued from Section 3 Model Architecture, Page 4]`) only when a large section is split across chunks.
4. **Location of Chunking Code**:
   - The production chunker belongs in `backend/chunking/chunker.py` (with `backend/chunking/__init__.py`), completely separated from tests and parser code.
   - Tests belong in `backend/tests/test_chunker.py`.
5. **Status of Chunker**:
   - The structure-aware chunker is **PLANNED, NOT IMPLEMENTED**.

---

## 8. Safety Constraints

- **No Unauthorized Model Calls**: Do not run Ollama (`qwen2.5:3b-instruct`) or any external model API.
- **No Unauthorized Pipeline Execution**: Do not execute OCR extraction, full summarization, script generation, or TTS generation.
- **Preserve Existing Outputs**: Existing files in `outputs/` are precious benchmark data. Never overwrite or delete them.
- **Parser Stability**: Do not modify `backend/document_parser/parser.py` unless a concrete, reproducible parsing defect is demonstrated.
- **Isolated Testing**: Any tests for new modules must use synthetic in-memory fixtures or `tempfile.TemporaryDirectory`.

---

## 9. Current Status and Next Planned Step

- **Current Status**: Project context, data flow, parser design, and architectural gaps have been analyzed and verified. Documentation phase active.
- **Next Planned Step**: Implement and independently test a reusable structure-aware chunker (`backend/chunking/chunker.py`) with isolated unit tests (`backend/tests/test_chunker.py`) using synthetic fixtures, **without** integrating it into the existing summarization pipeline or rerunning LLM calls.

---

## 10. Instructions for Future Coding Agents

### Mandatory Rules
1. **Read `PROJECT_CONTEXT.md` and `TASK_LOG.md`** at the start of every new session before making project changes.
2. **Never rerun OCR, summarization, audio-script generation, or TTS** without explicit user authorization.
3. **Never overwrite or delete existing outputs** without explicit authorization.
4. **Do not modify the parser** unless a specific issue is demonstrated and verified by tests.
5. **Do not claim planned work is complete** without verified implementation and test results.
6. **Keep changes minimal and test new functionality in isolation** using temporary or synthetic data.
7. **Ask for permission** before expanding the agreed scope.
