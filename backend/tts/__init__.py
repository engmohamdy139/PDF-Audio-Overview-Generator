"""Piper TTS synthesis package for dual-host dialogue narration."""

from backend.tts.mp3_converter import (
    DEFAULT_FFMPEG_BIN,
    convert_wav_to_mp3,
)
from backend.tts.piper_tts import (
    DEFAULT_PAUSE_DURATION,
    DEFAULT_PIPER_BIN,
    DEFAULT_PIPER_MODELS_DIR,
    DEFAULT_VOICE_ANALYST,
    DEFAULT_VOICE_EXPLAINER,
    combine_wav_files,
    commit_file_atomic,
    generate_piper_wav,
    resolve_model_paths,
    resolve_voice_for_speaker,
    synthesize_dialogue,
    validate_wav_format,
)

__all__ = [
    "DEFAULT_PIPER_BIN",
    "DEFAULT_PIPER_MODELS_DIR",
    "DEFAULT_VOICE_EXPLAINER",
    "DEFAULT_VOICE_ANALYST",
    "DEFAULT_PAUSE_DURATION",
    "DEFAULT_FFMPEG_BIN",
    "generate_piper_wav",
    "combine_wav_files",
    "commit_file_atomic",
    "resolve_model_paths",
    "resolve_voice_for_speaker",
    "synthesize_dialogue",
    "validate_wav_format",
    "convert_wav_to_mp3",
]
