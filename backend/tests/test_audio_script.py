
from pathlib import Path

import requests

SUMMARY_PATH = Path("outputs/summary.md")
OUTPUT_PATH = Path("outputs/audio_script.md")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b-instruct"


def generate_audio_script(summary: str) -> str:
    prompt = f"""
You are an educational audio script writer.

Transform the source summary into an engaging audio article
that sounds natural when read aloud.

Requirements:
- Write in clear, natural English.
- Start with an engaging introduction.
- Explain the main ideas step by step.
- Use smooth transitions between ideas.
- Explain technical terms in simple language.
- End with a concise recap and a memorable closing.
- Make the script engaging, like a podcast host speaking
  directly to a listener.
- Use only information supported by the source.
- Do not invent facts, experiments, or results.
- Do not include speaker labels, stage directions, or sound effects.
- Write approximately 400 to 600 words.
- Return only the final audio script, without commentary.

SOURCE SUMMARY:
{summary}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 4096,
            },
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Summary not found: {SUMMARY_PATH.resolve()}"
        )

    summary = SUMMARY_PATH.read_text(encoding="utf-8")

    if not summary.strip():
        raise ValueError("The summary file is empty.")

    print("Generating audio script with Qwen...")

    script = generate_audio_script(summary)

    if not script:
        raise ValueError("Qwen returned an empty audio script.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(script + "\n", encoding="utf-8")

    print("\n--- Generated Audio Script ---\n")
    print(script)
    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"File error: {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"Input/output error: {exc}")
        raise SystemExit(1)
    except requests.RequestException as exc:
        print(f"Ollama API error: {exc}")
        print("Check that Ollama is running and the model is available.")
        raise SystemExit(1)
