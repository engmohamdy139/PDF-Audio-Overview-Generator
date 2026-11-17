
from pathlib import Path
import sys

import requests


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import MODEL_NAME, NUM_CTX, OLLAMA_URL

SUMMARY_FILE = (
    BASE_DIR
    / "outputs"
    / "large_pdf"
    / "summary"
    / "final_summary.md"
)

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "large_pdf"
    / "audio_script.md"
)

def generate_audio_script(summary: str) -> str:
    prompt = f"""
You are an educational podcast scriptwriter.

Transform the research summary below into a natural English
audio script that is easy to listen to.

SOURCE FIDELITY:
- Use only information explicitly supported by the summary.
- Do not invent facts, examples, numbers, results, or conclusions.
- Preserve technical terms and numerical results accurately.
- If information is unclear or contradictory, omit it.
- Do not imply that you reviewed the original research paper.

AUDIO STYLE:
- Write as continuous spoken narration.
- Begin with a brief introduction to the research topic.
- Organize ideas in a logical sequence.
- Use natural transitions between ideas.
- Explain technical concepts only when the summary supports the explanation.
- Remove repeated findings and redundant information.
- Do not include section headings, bullet points, numbered lists,
  markdown formatting, or tables.
- Do not repeat the same result in different words.
- Finish with a short conclusion based only on the summary.
- Target approximately 500-700 words, but prioritize accuracy.

Return only the spoken script.

RESEARCH SUMMARY:
{summary}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": NUM_CTX,
            },
        },
        timeout=600,
    )

    response.raise_for_status()
    script = response.json()["response"].strip()

    if not script:
        raise ValueError("Qwen returned an empty audio script.")

    return script


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Summary file not found: {SUMMARY_FILE}"
        )

    summary = SUMMARY_FILE.read_text(encoding="utf-8").strip()

    if not summary:
        raise ValueError("The summary file is empty.")

    print(f"Summary length: {len(summary)} characters")
    print("Generating audio script with Qwen...")

    script = generate_audio_script(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(script + "\n", encoding="utf-8")

    print(f"\nAudio script saved to: {OUTPUT_FILE}")
    print(f"Script length: {len(script)} characters")
    print("\n--- SCRIPT PREVIEW ---\n")
    print(script[:2500])


if __name__ == "__main__":
    main()
