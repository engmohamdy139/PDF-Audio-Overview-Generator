# Session Handoff: PDF Audio Generator

## 1. Project Overview

- **Project Purpose**: The **PDF Audio Generator** converts technical PDF documents into structured summaries and conversational, podcast-style audio narration. The processing pipeline consists of:
  $$\text{PDF} \longrightarrow \text{OCR Extraction} \longrightarrow \text{Document Parsing} \longrightarrow \text{Structure-Aware Chunking} \longrightarrow \text{LLM Summarization} \longrightarrow \text{Audio Scriptwriting} \longrightarrow \text{TTS}$$
- **Current Environment & Working Directory**:
  - Operating System: Linux
  - Python Environment: Python 3.11 with `requests`, `python-dotenv`, `pydantic-settings`, `indra_ocr`, `gTTS`
  - Local LLM: Ollama running `qwen2.5:3b-instruct` (`num_ctx = 4096`)
  - Project Root: `/home/dev/projects/pdf_audio_generator/taskv1`
- **Reference Benchmark Document**:
  - Source PDF: `~/Downloads/Attention is all you need.pdf` (15 pages, Vaswani et al., 2017).

---

## 2. Verified Current State

The following components and changes have been implemented and independently verified in recent phases:

1. **Root-Level Environment Protection & Template**:
   - `.gitignore` was created at the project root, explicitly ignoring `.env`, `.env.*.local`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, and `.venv/`.
   - `.env.example` exists at the project root: `/home/dev/projects/pdf_audio_generator/taskv1/.env.example`, containing verified non-secret defaults (`OLLAMA_URL`, `MODEL_NAME`, `NUM_CTX`, `CHUNK_TARGET_SIZE`, `CHUNK_MAX_SIZE`, `TTS_LANGUAGE`, and commented `INDRA_OCR_MODEL_DIR`).
   - **`.env` Status**: `.env` has **NOT** been created and must remain uncreated for now. Do **not** run `cp .env.example .env` or create `.env` in any way. Creating `.env` will require separate explicit approval in a future phase.

2. **Centralized Configuration Module**:
   - `backend/config.py` was created to centralize project parameters with the clarified precedence order:
     1. Ambient system environment variables (highest precedence).
     2. Values loaded from a local `.env` via `python-dotenv` (`override=False`).
     3. In-code default values (fallback).
   - `backend/config.py` loads the root `.env` only if it exists; otherwise, it falls back gracefully to ambient environment variables and default values.
   - Verified via standalone import test (`python3 -c "from backend import config; print(config.get_config_dict())"`), confirming all default values load cleanly.

3. **Structure-Aware Chunker (Isolated Implementation)**:
   - `backend/chunking/__init__.py`, `backend/chunking/chunker.py`, and `backend/tests/test_chunker.py` were created.
   - Implements `DocumentChunk` and `chunk_document(document: ParsedDocument)`.
   - Preserves all blocks in reading order, coalesces small sections up to target size (5,000 chars), splits oversized sections between blocks with continuation headers (`[Context: Continued from ...]`), keeps HTML tables atomic, carries forward empty section titles, and handles oversized individual blocks via documented policies (`keep_atomic` and `split`).
   - Latest reported test result: **9 unit tests ran in 0.001s and passed (OK)** using synthetic in-memory fixtures.

4. **Phase 3A Centralized Config Integration (First Script)**:
   - `backend/tests/test_qwen.py` was updated to set up `sys.path` and import `OLLAMA_URL`, `MODEL_NAME`, and `NUM_CTX` from `backend.config`.
   - Replaced hardcoded `"num_ctx": 4096` with `"num_ctx": NUM_CTX`.
   - Verified clean syntax compilation with `py_compile`.

> [!IMPORTANT]
> **Verification Boundary**: Only the unit tests for `backend/chunking/` and the syntax/import checks for `backend/config.py` and `backend/tests/test_qwen.py` have been run. **The full end-to-end pipeline has NOT been executed or verified with the new components.** Do not claim that the full pipeline is tested or working.

---

## 3. Architecture and Implementation Boundaries

- **Role of `backend/config.py`**: A shared configuration module providing centralized access to Ollama URLs, model names, context window sizes, chunking limits, and TTS settings. It is passive and does not alter legacy script behavior until each script explicitly imports and uses it.
- **Scope of Configuration Integration**: Configuration centralization has been integrated **only into `backend/tests/test_qwen.py`** (Phase 3A). All other scripts (`test_large_pdf_summary.py`, `test_large_pdf_audio_script.py`, `test_pdf_summary.py`, `test_audio_script.py`, `test_tts.py`) still contain hardcoded values.
- **Chunker Integration Status**: The structure-aware chunker (`backend/chunking/chunker.py`) is fully implemented and tested in isolation, but **is NOT yet integrated into the legacy summarization script (`backend/tests/test_large_pdf_summary.py`)**. The legacy summarization script still contains its raw markdown page reader and naive character-level sliding window.

---

## 4. Existing Outputs That Must Be Preserved

The following generated files under `outputs/` represent significant compute and verified reference data. **Never overwrite, delete, truncate, or regenerate them without explicit user authorization**:

| Path | Description | Status |
| :--- | :--- | :--- |
| `outputs/large_pdf/pages/page_*.md` | 15 OCR Markdown pages (`page_0000.md` to `page_0014.md`) | **DO NOT MODIFY** |
| `outputs/large_pdf/state.json` | OCR extraction state file | **DO NOT MODIFY** |
| `outputs/large_pdf/parser/document_structure.json` | Serialized `ParsedDocument` structure | **DO NOT MODIFY** |
| `outputs/large_pdf/summary/chunks/chunk_*.md` | 9 legacy chunk summaries (`chunk_000.md` to `chunk_008.md`) | **DO NOT MODIFY** |
| `outputs/large_pdf/summary/state.json` | Legacy chunk summarization state file | **DO NOT MODIFY** |
| `outputs/large_pdf/summary/final_summary.md` | Reconciled comprehensive summary (8,853 bytes) | **DO NOT MODIFY** |
| `outputs/large_pdf/audio_script.md` | Educational podcast narration script (5,269 bytes) | **DO NOT MODIFY** |
| `outputs/summary.md` | 3-page prototype summary | **DO NOT MODIFY** |
| `outputs/audio_script.md` | 3-page prototype audio script | **DO NOT MODIFY** |
| `outputs/audio.mp3` | Generated narration MP3 audio (63,414 bytes) | **DO NOT MODIFY** |

---

## 5. Strict Working Rules for Future Sessions

1. **Read `SESSION_HANDOFF.md` first**, followed by `PROJECT_CONTEXT.md` and `TASK_LOG.md`.
2. **Resume directly from this handoff**: Inspect only specific, relevant files needed for the current task rather than re-reading or re-scanning the entire repository.
3. **Never execute pipelines or models without authorization**: Do not run OCR (`indra_ocr`), Ollama/Qwen model inference, summarization, script generation, TTS, or any end-to-end pipeline unless the user explicitly authorizes that exact run.
4. **Never overwrite or delete existing outputs**: All files in `outputs/` must remain untouched.
5. **Preserve the parser**: Do not modify `backend/document_parser/parser.py` unless a reproducible defect is demonstrated.
6. **No unauthorized dependencies**: Do not install packages or edit `requirements.txt` without approval.
7. **Scoped changes & diff proposals**: Make minimal changes. Always propose exact code diffs and await user approval before editing files.
8. **Evidence-based reporting**: Never claim planned work is complete or that full pipelines function without verified test execution.

---

## 6. Next Planned Phases (Not Yet Executed)

- **Phase 3B (Immediate Next Task)**:
  - Inspect `backend/tests/test_large_pdf_summary.py`.
  - Propose a minimal diff to replace its hardcoded `OLLAMA_URL`, `MODEL_NAME`, and `NUM_CTX` with imports from `backend.config`.
  - Wait for explicit user approval before making the edit.
  - Do **not** execute the script or run summarization.
- **Phase 3C**:
  - Review remaining scripts (`test_large_pdf_audio_script.py`, `test_pdf_summary.py`, `test_audio_script.py`, `test_tts.py`).
  - Propose and apply centralized configuration imports one script at a time, obtaining approval for each.
- **Phase 4**:
  - Review `backend/chunking/chunker.py` integration into `backend/tests/test_large_pdf_summary.py` (replacing `load_pages()` and `split_into_chunks()` with `parse_markdown_pages()` and `chunk_document()`).
  - Create an integration plan that protects existing outputs (e.g., using a separate output subdirectory or version tag).
  - Do not integrate or execute without explicit authorization.
- **Later Phases**:
  - Controlled end-to-end evaluation using non-colliding output paths under user supervision.

---

## 7. Resume Instructions for the Next Session

When starting a new session:
1. Read `SESSION_HANDOFF.md`, `PROJECT_CONTEXT.md`, and `TASK_LOG.md`.
2. Output a brief 3–4 line summary confirming the current status (Phase 1, Phase 2, and Phase 3A complete; `SESSION_HANDOFF.md` active).
3. Ask the user for permission to begin **Phase 3B** (inspecting `backend/tests/test_large_pdf_summary.py` and proposing a configuration-import diff).
4. **Do not start editing files automatically.** Wait for the user's explicit prompt before making any changes.
