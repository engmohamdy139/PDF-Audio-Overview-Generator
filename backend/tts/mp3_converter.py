"""Standalone WAV-to-MP3 audio converter for taskv1.

Converts uncompressed WAV audio files to MP3 using the local FFmpeg binary
and libmp3lame encoder, with race-safe atomic file commit protection.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import wave

from backend.tts.piper_tts import commit_file_atomic

DEFAULT_FFMPEG_BIN = Path(os.getenv("FFMPEG_BIN_PATH") or "/usr/bin/ffmpeg")
DEFAULT_MP3_BITRATE = "192k"
DEFAULT_CONVERSION_TIMEOUT = 120


def validate_bitrate(bitrate: str) -> str:
    """Validate and normalize the MP3 audio bitrate string.

    Args:
        bitrate: Bitrate specification (e.g. '128k', '192k', '256k', '320k').

    Returns:
        Normalized bitrate string suitable for FFmpeg '-b:a'.

    Raises:
        ValueError: If bitrate is invalid, non-string, or out of standard range (32k-320k).
    """
    if not isinstance(bitrate, str) or not bitrate.strip():
        raise ValueError(f"Bitrate must be a non-empty string, got {bitrate!r}.")

    clean = bitrate.strip().lower()

    # Match forms like "192k", "192kbps", "192 k"
    match_k = re.match(r"^(\d+)\s*(?:k|kbps|kbit/s)?$", clean)
    if match_k:
        val = int(match_k.group(1))
        # Handle bps values e.g. 192000
        if val >= 1000:
            val = val // 1000
        if val < 32 or val > 320:
            raise ValueError(
                f"Bitrate {bitrate!r} is out of standard MP3 range (32k to 320k)."
            )
        return f"{val}k"

    raise ValueError(
        f"Invalid bitrate specification: {bitrate!r}. "
        "Expected values such as '128k', '192k', '256k', or '320k'."
    )


def convert_wav_to_mp3(
    wav_path: str | Path,
    output_path: str | Path | None = None,
    bitrate: str = DEFAULT_MP3_BITRATE,
    ffmpeg_bin: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert an uncompressed WAV audio file to MP3 using FFmpeg and libmp3lame.

    Synthesizes the MP3 into a uniquely named temporary staging file in the target
    directory and atomically commits it. Pre-existing destination files are never
    modified or unlinked when overwrite is False. The source WAV is accessed strictly
    read-only and is never modified, deleted, moved, or overwritten.

    Args:
        wav_path: Path to the input WAV file.
        output_path: Optional destination MP3 path. If omitted, uses wav_path stem with '.mp3'.
        bitrate: Audio bitrate (default: '192k').
        ffmpeg_bin: Optional override path for the FFmpeg executable (default: /usr/bin/ffmpeg).
        overwrite: Whether to allow atomic replacement of an existing destination file.

    Returns:
        Path object pointing to the finished MP3 file.

    Raises:
        FileNotFoundError: If source WAV or FFmpeg binary does not exist.
        PermissionError: If FFmpeg binary is not executable.
        FileExistsError: If output_path exists and overwrite is False.
        ValueError: If input WAV is invalid, bitrate is invalid, or destination equals source.
        RuntimeError: If FFmpeg conversion fails or produces an empty output file.
    """
    # 1. Validate source WAV path and readability
    src_path = Path(wav_path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Source WAV file not found: {src_path}")
    if src_path.stat().st_size == 0:
        raise ValueError(f"Source WAV file is 0 bytes: {src_path}")

    try:
        with wave.open(str(src_path), "rb") as w:
            nframes = w.getnframes()
    except (wave.Error, EOFError) as err:
        raise ValueError(
            f"Source file is not a valid or readable WAV container: {src_path} ({err})"
        ) from err

    if nframes == 0:
        raise ValueError(f"Source WAV file contains 0 audio frames: {src_path}")

    # 2. Resolve and validate destination path
    if output_path is None:
        dest_path = src_path.with_suffix(".mp3")
    else:
        dest_path = Path(output_path).resolve()

    if dest_path == src_path:
        raise ValueError(f"Output path cannot be identical to input WAV path: {dest_path}")

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination MP3 file already exists: {dest_path}. "
            "Set overwrite=True to allow replacement."
        )

    # 3. Validate bitrate and FFmpeg executable
    clean_bitrate = validate_bitrate(bitrate)

    bin_path = Path(ffmpeg_bin or DEFAULT_FFMPEG_BIN).resolve()
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"FFmpeg executable not found at: {bin_path}. "
            "Ensure FFmpeg is installed or provide a valid ffmpeg_bin path."
        )
    if not os.access(bin_path, os.X_OK):
        raise PermissionError(f"FFmpeg binary at {bin_path} is not executable.")

    # 4. Create uniquely named temporary staging file in destination directory
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_stage = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=".mp3_stage_",
        suffix=".mp3",
        delete=False,
    )
    temp_stage_path = Path(temp_stage.name)
    temp_stage.close()

    cmd = [
        str(bin_path),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        clean_bitrate,
        str(temp_stage_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=DEFAULT_CONVERSION_TIMEOUT,
        )

        if not temp_stage_path.is_file() or temp_stage_path.stat().st_size == 0:
            raise RuntimeError(
                f"FFmpeg produced an empty or missing MP3 output file at {temp_stage_path}"
            )

        # Race-safe atomic commit to final destination
        commit_file_atomic(temp_stage_path, dest_path, overwrite=overwrite)
        return dest_path

    except subprocess.CalledProcessError as err:
        stderr_msg = err.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg MP3 conversion failed (exit code {err.returncode}): {stderr_msg}"
        ) from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            f"FFmpeg MP3 conversion timed out after {DEFAULT_CONVERSION_TIMEOUT}s for: {src_path}"
        ) from err
    finally:
        if temp_stage_path.is_file():
            temp_stage_path.unlink(missing_ok=True)
