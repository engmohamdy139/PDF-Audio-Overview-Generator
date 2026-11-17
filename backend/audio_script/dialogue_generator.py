"""Single LLM Two-Host Dialogue Script Generator for taskv1.

Generates structured, multi-turn podcast dialogue between an Explainer (speaker-1)
and a Curious Analyst (speaker-2) from a research summary using a single Ollama call.
"""

from __future__ import annotations

import json
import re
from typing import Any
import urllib.error
import urllib.request

from backend.config import MODEL_NAME, NUM_CTX, OLLAMA_URL

# ---------------------------------------------------------------------------
# Dialogue JSON Schema Definition (Ollama Structured Output)
# ---------------------------------------------------------------------------

DIALOGUE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dialogue": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {
                        "type": "string",
                        "enum": ["speaker-1", "speaker-2"],
                    },
                    "text": {
                        "type": "string",
                    },
                },
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
            "minItems": 8,
            "maxItems": 14,
        },
    },
    "required": ["dialogue"],
    "additionalProperties": False,
}

# Regex to detect prohibited leading speaker prefixes in spoken dialogue text
PROHIBITED_PREFIX_PATTERN = re.compile(
    r"^(?:speaker[-_ ]?[12]|host[-_ ]?[12]|explainer|curious[-_ ]?analyst|analyst|co[- ]?host|professor|student)\s*:\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_dialogue_prompt(summary_text: str) -> str:
    """Construct the prompt instructing Ollama to generate a structured two-host dialogue.

    Args:
        summary_text: The source technical research summary.

    Returns:
        Formatted prompt string.
    """
    return f"""You are an expert dialogue writer for an educational science and technology podcast.
Transform the research summary provided below into an engaging, natural two-host dialogue.

HOST ROLES:
- speaker-1 (Explainer): Leads the explanation, introduces core concepts clearly and progressively, explains architectural and technical details, and connects ideas into a coherent narrative.
- speaker-2 (Curious Analyst): Provides meaningful analysis, asks thoughtful probing questions, challenges unclear points, draws analogies, and helps clarify practical implications and limitations.

CONVERSATION RULES:
1. Opening Speaker: The dialogue MUST start with speaker-1 introducing the research topic and core problem.
2. Strict Alternation: Alternate speakers strictly on every turn (speaker-1, speaker-2, speaker-1, speaker-2, ...). Never assign consecutive turns to the same speaker.
3. Turn Count: Produce between 8 and 14 dialogue turns in total (target 10 to 12 turns).
4. Turn Length & Substantive Contributions:
   - Most turns should contain approximately 25 to 55 words.
   - Concise but meaningful analytical questions or reactions may contain 12 to 24 words.
   - Hard constraint: Every turn MUST contain at least 12 words and no more than 75 words.
   - Every turn must be a complete, meaningful contribution. Do not add filler or unnecessary wording just to increase word count.
   - AVOID one-word acknowledgements or superficial filler.
   - No turn may be a monologue exceeding 75 words.
5. Self-Verification: Before outputting the final JSON, internally verify that every single turn contains at least 12 words and no more than 75 words.
6. Natural Conversational Flow: Write authentic, dynamic dialogue. Avoid repetitive greetings, artificial agreement, or redundant recap summaries.
7. Factual Accuracy & Source Fidelity: Preserve technical terminology and numerical results accurately. Ground the entire discussion strictly in the provided research summary. Do NOT introduce or invent external benchmarks, dates, co-authors, or unsupported claims.
8. Clean Spoken Text: Do NOT include speaker labels, prefixes, or stage directions inside the 'text' field (NEVER write 'speaker-1:', 'Host:', or '[laughs]'). Every word in 'text' will be read aloud by Text-to-Speech verbatim.
9. Output Format: Output ONLY valid JSON matching the required schema. Do not include introductory text, markdown explanations, or trailing commentary.

RESEARCH SUMMARY:
{summary_text.strip()}
"""


# ---------------------------------------------------------------------------
# Dialogue Structure Validation
# ---------------------------------------------------------------------------

def validate_dialogue_structure(
    dialogue: list[dict[str, str]],
    min_turns: int = 8,
    max_turns: int = 14,
    min_words_per_turn: int = 12,
    max_words_per_turn: int = 75,
) -> None:
    """Validate dialogue against two-host conversational and structural rules.

    Args:
        dialogue: List of dialogue turn dictionaries.
        min_turns: Minimum allowable turn count (inclusive).
        max_turns: Maximum allowable turn count (inclusive).
        min_words_per_turn: Minimum words allowed per spoken turn.
        max_words_per_turn: Maximum words allowed per spoken turn.

    Raises:
        ValueError: If dialogue format, speakers, alternation, word count, or prefixes are invalid.
    """
    if not isinstance(dialogue, list):
        raise ValueError(f"Dialogue must be a list, got {type(dialogue).__name__}.")

    turn_count = len(dialogue)
    if not (min_turns <= turn_count <= max_turns):
        raise ValueError(
            f"Dialogue turn count {turn_count} is out of bounds; expected between {min_turns} and {max_turns} turns."
        )

    for i, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {i + 1} must be a dictionary, got {type(turn).__name__}.")

        turn_keys = set(turn.keys())
        if turn_keys != {"speaker", "text"}:
            raise ValueError(
                f"Turn {i + 1} must contain exactly 'speaker' and 'text' keys, got: {sorted(turn_keys)}"
            )

        speaker = turn["speaker"]
        text = turn["text"]

        if not isinstance(speaker, str) or speaker not in ("speaker-1", "speaker-2"):
            raise ValueError(
                f"Turn {i + 1} has invalid speaker {speaker!r}; expected 'speaker-1' or 'speaker-2'."
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Turn {i + 1} contains empty or whitespace-only text.")

        if PROHIBITED_PREFIX_PATTERN.match(text.strip()):
            raise ValueError(
                f"Turn {i + 1} contains a prohibited leading speaker prefix: '{text[:25]}...'"
            )

        word_count = len(text.strip().split())
        if not (min_words_per_turn <= word_count <= max_words_per_turn):
            raise ValueError(
                f"Turn {i + 1} ({speaker}) has {word_count} words; "
                f"expected between {min_words_per_turn} and {max_words_per_turn} words."
            )

    if dialogue[0]["speaker"] != "speaker-1":
        raise ValueError(
            f"Dialogue must begin with 'speaker-1' (Explainer), but started with '{dialogue[0]['speaker']}'."
        )

    for i in range(1, turn_count):
        if dialogue[i]["speaker"] == dialogue[i - 1]["speaker"]:
            raise ValueError(
                f"Strict alternation violated at turn {i + 1}: "
                f"consecutive turns assigned to '{dialogue[i]['speaker']}'."
            )


# ---------------------------------------------------------------------------
# Core Dialogue Generation
# ---------------------------------------------------------------------------

def generate_dialogue_script(
    summary_text: str,
    model_name: str | None = None,
    ollama_url: str | None = None,
    num_ctx: int | None = None,
    temperature: float = 0.3,
    timeout_seconds: int = 300,
) -> list[dict[str, str]]:
    """Generate a structured two-host podcast dialogue using a single Ollama LLM call.

    Args:
        summary_text: Input research summary text to base the dialogue upon.
        model_name: Optional override for the Ollama model (defaults to config.MODEL_NAME).
        ollama_url: Optional override for the Ollama generate URL (defaults to config.OLLAMA_URL).
        num_ctx: Optional override for context window size (defaults to config.NUM_CTX).
        temperature: Sampling temperature for generation (default: 0.3).
        timeout_seconds: HTTP request timeout in seconds (default: 300).

    Returns:
        Validated list of dialogue turn dicts: [{"speaker": "speaker-1", "text": "..."}, ...]

    Raises:
        ValueError: If summary_text is invalid, JSON parsing fails, or dialogue validation fails.
        RuntimeError: If the Ollama HTTP request fails or returns an invalid envelope.
    """
    if not isinstance(summary_text, str) or not summary_text.strip():
        raise ValueError("summary_text must be a non-empty string.")

    effective_model = model_name or MODEL_NAME
    effective_url = ollama_url or OLLAMA_URL
    effective_num_ctx = num_ctx if num_ctx is not None else NUM_CTX

    prompt = build_dialogue_prompt(summary_text)

    payload = {
        "model": effective_model,
        "prompt": prompt,
        "format": DIALOGUE_JSON_SCHEMA,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(effective_num_ctx),
            "num_predict": 1500,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        effective_url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise RuntimeError(f"Ollama returned non-200 HTTP status code: {status_code}")
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama HTTP error {err.code}: {err_body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to connect to Ollama at {effective_url}: {err.reason}") from err
    except TimeoutError as err:
        raise RuntimeError(f"Ollama request timed out after {timeout_seconds}s.") from err

    try:
        envelope = json.loads(raw_body)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Malformed JSON in Ollama response envelope: {err}") from err

    if not isinstance(envelope, dict):
        raise RuntimeError(f"Expected dict response envelope from Ollama, got {type(envelope).__name__}.")

    raw_content = envelope.get("response")
    if raw_content is None or not isinstance(raw_content, str):
        raise RuntimeError("Ollama response envelope is missing a valid string 'response' field.")

    clean_content = raw_content.strip()
    if clean_content.startswith("```"):
        clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r"\s*```$", "", clean_content)
        clean_content = clean_content.strip()

    try:
        data = json.loads(clean_content)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Failed to parse generated dialogue as JSON: {err}. Raw output was: {raw_content[:200]!r}"
        ) from err

    if not isinstance(data, dict):
        raise ValueError(f"Dialogue JSON root must be an object, got {type(data).__name__}.")

    if set(data.keys()) != {"dialogue"}:
        raise ValueError(
            f"Dialogue JSON root must contain only 'dialogue' key, got keys: {sorted(data.keys())}"
        )

    dialogue = data["dialogue"]
    validate_dialogue_structure(dialogue)

    return dialogue
