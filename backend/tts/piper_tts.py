"""Standalone Piper TTS speech synthesis engine for taskv1.

Uses the bundled Piper C++ runtime and ONNX voice models
stored inside the project.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import wave

# ---------------------------------------------------------------------------
# Path & Configuration Setup (Self-Contained & Relative)
# ---------------------------------------------------------------------------

# Resolve Piper runtime and voice models from inside taskv1.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPER_TOOLS_DIR = PROJECT_ROOT / "tools" / "piper"

DEFAULT_PIPER_BIN = Path(
    os.getenv("PIPER_BIN_PATH")
    or (PIPER_TOOLS_DIR / "piper" / "piper")
)
DEFAULT_PIPER_MODELS_DIR = Path(
    os.getenv("PIPER_MODELS_DIR")
    or (PIPER_TOOLS_DIR / "models")
)

# Default voice models for Host 1 (Explainer) and Host 2 (Curious Analyst)
DEFAULT_VOICE_EXPLAINER = os.getenv("PIPER_VOICE_EXPLAINER", "en_US-lessac-medium")
DEFAULT_VOICE_ANALYST = os.getenv("PIPER_VOICE_ANALYST", "en_US-hfc_female-medium")
DEFAULT_PAUSE_DURATION = float(os.getenv("PIPER_PAUSE_DURATION", "0.3"))

# Verified audio container constraints for Piper ONNX medium models
EXPECTED_CHANNELS = 1
EXPECTED_SAMPWIDTH = 2  # 16-bit PCM (2 bytes per sample)
EXPECTED_FRAMERATE = 22050
EXPECTED_COMPTYPE = "NONE"

# Buffer size for streaming WAV frame merging (4,096 frames = ~8 KB at 22.05 kHz 16-bit mono)
CHUNK_FRAME_SIZE = 4096

# Regex to strip accidental conversational speaker prefixes before speech generation
SPEAKER_PREFIX_PATTERN = re.compile(
    r"^(?:speaker[-_ ]?[12]|host[-_ ]?[12]|explainer|analyst|curious[-_ ]?analyst|co[- ]?host|professor|student)\s*:\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Atomic File Operations & Validation Helpers
# ---------------------------------------------------------------------------

def commit_file_atomic(
    temp_path: Path,
    target_path: Path,
    overwrite: bool = False,
) -> None:
    """Atomically commit temp_path to target_path with race-safe no-overwrite protection.

    On Linux, when overwrite is False, uses os.link() to create a hard link. The kernel
    system call fails atomically with FileExistsError if target_path already exists,
    guaranteeing that pre-existing destination files are never modified, truncated, or unlinked.
    When overwrite is True, uses os.replace() for atomic POSIX replacement.

    Args:
        temp_path: Staged temporary source file.
        target_path: Destination target path (must be on the same filesystem).
        overwrite: Whether to allow atomic replacement of an existing destination file.

    Raises:
        FileNotFoundError: If temp_path does not exist.
        FileExistsError: If target_path exists and overwrite is False.
    """
    if not temp_path.is_file():
        raise FileNotFoundError(f"Temporary staging file not found: {temp_path}")

    if not overwrite:
        try:
            os.link(temp_path, target_path)
        except FileExistsError as err:
            raise FileExistsError(
                f"Destination file already exists: {target_path}. "
                "Atomic commit aborted without modifying pre-existing file. "
                "Set overwrite=True to allow replacement."
            ) from err
        temp_path.unlink(missing_ok=True)
    else:
        os.replace(temp_path, target_path)


def resolve_model_paths(
    voice_name: str,
    models_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Validate and return (model_path, config_path) for a voice model.

    Args:
        voice_name: Voice model identifier (e.g. 'en_US-lessac-medium').
        models_dir: Optional override directory containing ONNX models.

    Returns:
        Tuple of (model_onnx_path, config_json_path).

    Raises:
        ValueError: If voice_name is empty or whitespace-only.
        FileNotFoundError: If models directory or model/config files do not exist.
    """
    clean_voice = str(voice_name).strip()
    if not clean_voice:
        raise ValueError("voice_name cannot be empty or whitespace-only.")

    base_voice = clean_voice[:-5] if clean_voice.endswith(".onnx") else clean_voice
    target_models_dir = Path(models_dir or DEFAULT_PIPER_MODELS_DIR).resolve()

    if not target_models_dir.is_dir():
        raise FileNotFoundError(f"Piper models directory not found: {target_models_dir}")

    model_path = target_models_dir / f"{base_voice}.onnx"
    config_path = target_models_dir / f"{base_voice}.onnx.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Piper ONNX model not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Piper voice config not found: {config_path}")

    return model_path, config_path


def validate_wav_format(wav_path: Path, label: str = "WAV file") -> tuple[int, int, int]:
    """Validate that a WAV file exists, is non-empty, and matches 22,050 Hz 16-bit mono PCM.

    Args:
        wav_path: Path to the WAV file to validate.
        label: Descriptive label for error reporting.

    Returns:
        Tuple of (nchannels, sampwidth, framerate).

    Raises:
        RuntimeError: If file is missing, empty, or not a valid WAV container.
        ValueError: If audio parameters do not match 22,050 Hz 16-bit mono PCM.
    """
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing or 0 bytes: {wav_path}")

    try:
        with wave.open(str(wav_path), "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            comptype = w.getcomptype()
            nframes = w.getnframes()
    except wave.Error as err:
        raise RuntimeError(f"{label} is not a valid WAV container: {err}") from err

    if comptype != EXPECTED_COMPTYPE:
        raise ValueError(
            f"{label} has unsupported compression type '{comptype}'; "
            f"expected uncompressed PCM ('{EXPECTED_COMPTYPE}')."
        )
    if nchannels != EXPECTED_CHANNELS:
        raise ValueError(
            f"{label} has {nchannels} channel(s); expected {EXPECTED_CHANNELS} (mono)."
        )
    if sampwidth != EXPECTED_SAMPWIDTH:
        raise ValueError(
            f"{label} has {sampwidth * 8}-bit sample width; "
            f"expected {EXPECTED_SAMPWIDTH * 8}-bit (16-bit PCM)."
        )
    if framerate != EXPECTED_FRAMERATE:
        raise ValueError(
            f"{label} has sample rate {framerate} Hz; expected {EXPECTED_FRAMERATE} Hz."
        )
    if nframes == 0:
        raise RuntimeError(f"{label} contains 0 audio frames: {wav_path}")

    return nchannels, sampwidth, framerate


# ---------------------------------------------------------------------------
# Core Speech Synthesis
# ---------------------------------------------------------------------------

def generate_piper_wav(
    text: str,
    voice_name: str,
    output_path: str | Path,
    piper_bin: Path | None = None,
    models_dir: Path | None = None,
    length_scale: float = 1.0,
    timeout_seconds: int = 60,
    overwrite: bool = False,
) -> Path:
    """Synthesize a single dialogue line to a 16-bit PCM WAV file using Piper.

    Synthesizes into an isolated temporary staging file in dest_path.parent and
    atomically commits. Pre-existing files at output_path are never modified or
    unlinked if synthesis fails.

    Args:
        text: Plain-text string to synthesize into speech.
        voice_name: Model identifier (e.g. 'en_US-lessac-medium').
        output_path: Destination WAV file path.
        piper_bin: Optional override for the Piper executable path.
        models_dir: Optional override for the directory containing ONNX models.
        length_scale: Phoneme duration multiplier (1.0 = normal speaking tempo).
        timeout_seconds: Subprocess execution timeout in seconds.
        overwrite: Whether to allow atomic replacement of an existing destination file.

    Returns:
        Path object pointing to the successfully created WAV file.

    Raises:
        FileExistsError: If output_path exists and overwrite is False.
        FileNotFoundError: If Piper binary or voice models are not found.
        ValueError: If input text is empty or whitespace-only.
        RuntimeError: If Piper process fails or generates an invalid WAV.
    """
    dest_path = Path(output_path).resolve()
    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination WAV file already exists: {dest_path}. "
            "Set overwrite=True to allow replacement."
        )

    bin_path = Path(piper_bin or DEFAULT_PIPER_BIN).resolve()
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"Piper executable not found at: {bin_path}. "
            f"Ensure bundled Piper tools are available at {PIPER_TOOLS_DIR}."
        )

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Cannot synthesize speech for empty or whitespace-only text.")

    sanitized_text = SPEAKER_PREFIX_PATTERN.sub("", clean_text).strip()
    if not sanitized_text:
        sanitized_text = clean_text

    model_path, config_path = resolve_model_paths(voice_name, models_dir)
    base_voice = model_path.stem

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary staging file in the same directory to allow atomic hard-link commit
    temp_stage = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=".piper_turn_stage_",
        suffix=".wav",
        delete=False,
    )
    temp_stage_path = Path(temp_stage.name)
    temp_stage.close()

    # Ensure dynamic libraries in the piper folder are discoverable
    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_path.parent}:{existing_ld}".rstrip(":")

    cmd = [
        str(bin_path),
        "--model",
        str(model_path),
        "--config",
        str(config_path),
        "--length_scale",
        str(length_scale),
        "--output_file",
        str(temp_stage_path),
    ]

    try:
        subprocess.run(
            cmd,
            input=sanitized_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout_seconds,
            env=env,
        )

        validate_wav_format(temp_stage_path, label=f"Synthesized turn WAV ('{base_voice}')")
        commit_file_atomic(temp_stage_path, dest_path, overwrite=overwrite)
        return dest_path

    except subprocess.CalledProcessError as err:
        stderr_msg = err.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Piper process failed (exit code {err.returncode}) for voice '{base_voice}': {stderr_msg}"
        ) from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            f"Piper synthesis timed out after {timeout_seconds}s for text: '{sanitized_text[:50]}...'"
        ) from err
    finally:
        if temp_stage_path.is_file():
            temp_stage_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Audio Merging
# ---------------------------------------------------------------------------

def combine_wav_files(
    wav_files: Sequence[str | Path],
    output_path: str | Path,
    pause_duration: float = DEFAULT_PAUSE_DURATION,
    overwrite: bool = False,
) -> Path:
    """Concatenate multiple WAV files sequentially using chunk-based streaming.

    Streams audio frames in 4,096-frame chunks using standard library wave,
    keeping memory usage flat regardless of total duration. Staged atomically
    to prevent corrupt or partially written output files.

    Args:
        wav_files: Ordered list of WAV file paths to concatenate.
        output_path: Destination path for the merged master WAV file.
        pause_duration: Duration of silence (in seconds) between dialogue turns.
        overwrite: Whether to allow atomic replacement of an existing destination file.

    Returns:
        Path object pointing to the combined WAV file.

    Raises:
        FileExistsError: If output_path exists and overwrite is False.
        ValueError: If wav_files is empty, pause_duration is invalid, or WAV parameters mismatch.
        FileNotFoundError: If any input WAV file is missing.
    """
    dest_path = Path(output_path).resolve()
    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination merged WAV file already exists: {dest_path}. "
            "Set overwrite=True to allow replacement."
        )

    if (
        not isinstance(pause_duration, (int, float))
        or not math.isfinite(pause_duration)
        or pause_duration < 0.0
    ):
        raise ValueError(
            f"pause_duration must be a non-negative finite number (seconds), got {pause_duration!r}"
        )

    if not wav_files:
        raise ValueError("Cannot combine audio: wav_files sequence is empty.")

    file_paths = [Path(p).resolve() for p in wav_files]
    for p in file_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Input WAV segment not found: {p}")

    # Validate all segments against 22,050 Hz mono 16-bit PCM before merging
    for p in file_paths:
        validate_wav_format(p, label=f"Input segment '{p.name}'")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_stage = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=".combine_stage_",
        suffix=".wav",
        delete=False,
    )
    temp_stage_path = Path(temp_stage.name)
    temp_stage.close()

    try:
        with wave.open(str(temp_stage_path), "wb") as out_wav:
            out_wav.setnchannels(EXPECTED_CHANNELS)
            out_wav.setsampwidth(EXPECTED_SAMPWIDTH)
            out_wav.setframerate(EXPECTED_FRAMERATE)

            silence_frames = max(0, int(EXPECTED_FRAMERATE * pause_duration))
            silence_bytes = bytes(silence_frames * EXPECTED_SAMPWIDTH * EXPECTED_CHANNELS)

            for i, wav_path in enumerate(file_paths):
                with wave.open(str(wav_path), "rb") as in_wav:
                    while True:
                        frames = in_wav.readframes(CHUNK_FRAME_SIZE)
                        if not frames:
                            break
                        out_wav.writeframes(frames)

                if i < len(file_paths) - 1 and silence_bytes:
                    out_wav.writeframes(silence_bytes)

        # Validate staged merged WAV before committing
        validate_wav_format(temp_stage_path, label="Merged master WAV")
        commit_file_atomic(temp_stage_path, dest_path, overwrite=overwrite)
        return dest_path

    finally:
        if temp_stage_path.is_file():
            temp_stage_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Speaker Resolution & Dialogue Orchestration
# ---------------------------------------------------------------------------

def resolve_voice_for_speaker(
    speaker: str,
    voice_map: Mapping[str, str] | None = None,
) -> str:
    """Resolve speaker name/role to a Piper voice model identifier.

    Args:
        speaker: Speaker label (e.g. 'Host 1', 'Explainer', 'Speaker 2', 'Curious Analyst').
        voice_map: Optional dictionary providing explicit speaker-to-voice mappings.

    Returns:
        Voice model identifier string.

    Raises:
        TypeError: If voice_map is provided but is not a Mapping.
        ValueError: If speaker is empty, voice_map contains empty keys/values,
                    or speaker cannot be mapped to a known voice.
    """
    if voice_map is not None:
        if not isinstance(voice_map, Mapping):
            raise TypeError(f"voice_map must be a mapping, got {type(voice_map).__name__}")
        for k, v in voice_map.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("voice_map contains an empty or whitespace-only speaker key.")
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"voice_map contains an empty voice value for speaker key {k!r}.")

    clean_speaker = str(speaker).strip()
    if not clean_speaker:
        raise ValueError("Speaker identifier cannot be empty or whitespace-only.")

    # Check explicit user mapping first (exact and canonical)
    if voice_map and clean_speaker in voice_map:
        return voice_map[clean_speaker].strip()

    canonical = re.sub(r"[^a-z0-9]", "", clean_speaker.lower())
    if voice_map:
        for k, v in voice_map.items():
            if re.sub(r"[^a-z0-9]", "", k.lower()) == canonical:
                return v.strip()

    # Host 1 / Explainer aliases
    if canonical in ("speaker1", "host1", "explainer", "host", "narrator"):
        return DEFAULT_VOICE_EXPLAINER

    # Host 2 / Curious Analyst aliases
    if canonical in ("speaker2", "host2", "curiousanalyst", "analyst", "cohost"):
        return DEFAULT_VOICE_ANALYST

    raise ValueError(
        f"Unrecognized speaker '{speaker}'. Must map to Host 1 (Explainer) or "
        f"Host 2 (Curious Analyst), or be provided in voice_map."
    )


def synthesize_dialogue(
    dialogue_turns: Sequence[Mapping[str, Any] | tuple[Any, ...]],
    output_path: str | Path,
    voice_map: Mapping[str, str] | None = None,
    pause_duration: float = DEFAULT_PAUSE_DURATION,
    piper_bin: Path | None = None,
    models_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Synthesize multi-turn dialogue into an output WAV with race-safe atomic commit and cleanup.

    Turns can be formatted as:
      - 2-tuples: ("Host 1", "...")
      - dicts: {"speaker": "Host 1", "text": "..."}
      - objects with .speaker and .text attributes (e.g. Pydantic models)

    Validates all turns, speakers, voices, and ONNX model files upfront before
    executing speech generation. Temporary turn WAV files use predictable indexed
    filenames ('turn_000.wav') independent of voice names.

    The merged WAV is written to an atomic staging file in dest_path.parent and committed.
    If synthesis fails at any point, all temporary files are cleaned up and any pre-existing
    destination file remains untouched.

    Args:
        dialogue_turns: Ordered sequence of dialogue turns.
        output_path: Destination path for the master WAV file.
        voice_map: Optional dict mapping speaker names to voice identifiers.
        pause_duration: Seconds of silence between turns.
        piper_bin: Optional override for the Piper binary path.
        models_dir: Optional override for the voice models directory.
        overwrite: Whether to allow atomic replacement of an existing destination file.

    Returns:
        Path pointing to the finished master WAV file.

    Raises:
        FileExistsError: If output_path exists and overwrite is False.
        ValueError: If dialogue_turns is empty or contains malformed turns.
        RuntimeError: If any turn synthesis or audio merging fails.
    """
    dest_path = Path(output_path).resolve()
    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination dialogue WAV file already exists: {dest_path}. "
            "Set overwrite=True to allow replacement."
        )

    if (
        not isinstance(pause_duration, (int, float))
        or not math.isfinite(pause_duration)
        or pause_duration < 0.0
    ):
        raise ValueError(
            f"pause_duration must be a non-negative finite number (seconds), got {pause_duration!r}"
        )

    if not dialogue_turns:
        raise ValueError("Cannot synthesize dialogue: dialogue_turns sequence is empty.")

    bin_path = Path(piper_bin or DEFAULT_PIPER_BIN).resolve()
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"Piper executable not found at: {bin_path}. "
            f"Ensure bundled Piper tools are available at {PIPER_TOOLS_DIR}."
        )

    # 1. Upfront validation of all dialogue turns, speakers, voices, and ONNX model files
    validated_turns: list[tuple[str, str, str]] = []
    for i, turn in enumerate(dialogue_turns):
        if isinstance(turn, tuple):
            if len(turn) != 2:
                raise ValueError(
                    f"Dialogue turn tuple at index {i} must contain exactly 2 elements (speaker, text), "
                    f"got {len(turn)} elements: {turn!r}"
                )
            speaker, text = turn[0], turn[1]
        elif isinstance(turn, Mapping):
            if "speaker" not in turn:
                raise ValueError(
                    f"Dialogue turn mapping at index {i} is missing required 'speaker' key: {turn!r}"
                )
            if "text" not in turn:
                raise ValueError(
                    f"Dialogue turn mapping at index {i} is missing required 'text' key: {turn!r}"
                )
            speaker, text = turn["speaker"], turn["text"]
        elif hasattr(turn, "speaker") and hasattr(turn, "text"):
            speaker, text = turn.speaker, turn.text
        else:
            raise ValueError(
                f"Invalid dialogue turn format at index {i}: {type(turn).__name__}. "
                "Expected mapping with 'speaker' and 'text', 2-tuple (speaker, text), "
                "or object with .speaker and .text attributes."
            )

        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(
                f"Dialogue turn at index {i} has an invalid or empty speaker: {speaker!r}"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Dialogue turn at index {i} (speaker '{speaker}') contains empty or whitespace-only text."
            )

        voice = resolve_voice_for_speaker(speaker, voice_map)
        resolve_model_paths(voice, models_dir)
        validated_turns.append((speaker, text, voice))

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    turn_wav_paths: list[Path] = []
    temp_dir = tempfile.TemporaryDirectory(prefix="taskv1_piper_")

    # Staging master file in destination parent directory for atomic hard-link commit
    temp_master = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=".dialogue_stage_",
        suffix=".wav",
        delete=False,
    )
    temp_master_path = Path(temp_master.name)
    temp_master.close()

    try:
        temp_dir_path = Path(temp_dir.name)

        # 2. Sequential synthesis using predictable indexed filenames independent of voice names
        for i, (speaker, text, voice) in enumerate(validated_turns):
            turn_file_path = temp_dir_path / f"turn_{i:03d}.wav"

            generate_piper_wav(
                text=text,
                voice_name=voice,
                output_path=turn_file_path,
                piper_bin=piper_bin,
                models_dir=models_dir,
                overwrite=True,
            )
            turn_wav_paths.append(turn_file_path)

        # 3. Stream-merge into atomic staging file
        combine_wav_files(
            wav_files=turn_wav_paths,
            output_path=temp_master_path,
            pause_duration=pause_duration,
            overwrite=True,
        )

        # 4. Final audio format validation on staged output
        validate_wav_format(temp_master_path, label="Merged dialogue master WAV")

        # 5. Race-safe atomic commit to destination
        commit_file_atomic(temp_master_path, dest_path, overwrite=overwrite)
        return dest_path

    finally:
        temp_dir.cleanup()
        if temp_master_path.is_file():
            temp_master_path.unlink(missing_ok=True)
