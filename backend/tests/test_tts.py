
from pathlib import Path

from gtts import gTTS

INPUT_PATH = Path("outputs/audio_script.md")
OUTPUT_PATH = Path("outputs/audio.mp3")

LANGUAGE = "en"
SLOW = False


def generate_audio(text: str) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tts = gTTS(
        text=text,
        lang=LANGUAGE,
        slow=SLOW,
        timeout=(10, 60),
    )

    tts.save(str(OUTPUT_PATH))


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Audio script not found: {INPUT_PATH.resolve()}"
        )

    text = INPUT_PATH.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("Audio script is empty.")

    print("Generating English audio with gTTS...")
    generate_audio(text)

    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError("Audio file was not generated correctly.")

    print(f"Audio saved to: {OUTPUT_PATH.resolve()}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"File error: {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"Input error: {exc}")
        raise SystemExit(1)
    except (OSError, RuntimeError) as exc:
        print(f"Audio generation error: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        raise SystemExit(1)
