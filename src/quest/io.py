from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .model import Passage


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("each JSONL row must be an object")
            yield value


def load_passages(path: str | Path) -> list[Passage]:
    rows = []
    for index, value in enumerate(read_jsonl(path)):
        rows.append(
            Passage(
                id=str(value.get("id", index)),
                title=str(value.get("title", "")),
                text=str(value["text"]),
            )
        )
    return rows


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def answer_to_dict(result) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "context": [asdict(passage) for passage in result.context.passages],
        "diagnostics": result.diagnostics,
    }
