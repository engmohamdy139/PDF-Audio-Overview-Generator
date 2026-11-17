"""Dialogue script generation package for dual-host podcast narration."""

from backend.audio_script.dialogue_generator import (
    DIALOGUE_JSON_SCHEMA,
    build_dialogue_prompt,
    generate_dialogue_script,
    validate_dialogue_structure,
)

__all__ = [
    "DIALOGUE_JSON_SCHEMA",
    "build_dialogue_prompt",
    "generate_dialogue_script",
    "validate_dialogue_structure",
]
