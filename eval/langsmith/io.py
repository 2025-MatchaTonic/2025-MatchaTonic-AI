from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.langsmith.schemas import validate_dataset_example


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
        validate_dataset_example(example, line_no=line_no)
        examples.append(example)
    if not examples:
        raise ValueError(f"{path} has no examples")
    return examples
