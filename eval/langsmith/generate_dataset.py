from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


CHAT_COMPLETIONS_ONLY_HINT = (
    "The selected model is not supported by the chat-completions endpoint. "
    "Use a chat-capable model such as gpt-4o, gpt-4o-mini, or the repo's OPENAI_MODEL, "
    "or use an OpenAI model id that is available to your API account."
)


def _response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate JSONL dataset examples with an LLM.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--prompt", default="eval/prompts/generate_team_project_dataset.md")
    parser.add_argument("--output", default="eval/datasets/generated/team_project_v1.jsonl")
    args = parser.parse_args()

    prompt = Path(args.prompt).read_text(encoding="utf-8").replace("{N}", str(args.count))
    client = OpenAI()

    request = {
        "model": args.model,
        "input": prompt,
        "temperature": 0.4,
    }
    try:
        response = client.responses.create(**request)
    except Exception as exc:
        if "temperature" not in str(exc):
            raise RuntimeError(f"{exc}\n\n{CHAT_COMPLETIONS_ONLY_HINT}") from exc
        request.pop("temperature", None)
        response = client.responses.create(**request)

    content = _response_text(response)
    if not content:
        raise RuntimeError("OpenAI response did not contain text output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote dataset candidate to {output_path}")


if __name__ == "__main__":
    main()
