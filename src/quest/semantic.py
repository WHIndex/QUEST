from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence


Vector = Sequence[float]
Embedder = Callable[[Sequence[str]], Sequence[Vector]]


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def normalize_text(value: str) -> str:
    return " ".join(_TOKEN.findall(value.lower()))


def lexical_tokens(value: str) -> set[str]:
    return set(normalize_text(value).split())


def lexical_similarity(left: str, right: str) -> float:
    a = lexical_tokens(left)
    b = lexical_tokens(right)
    if not a or not b:
        return 0.0
    overlap = len(a & b) / len(a | b)
    containment = len(a & b) / min(len(a), len(b))
    exact = 1.0 if normalize_text(left) == normalize_text(right) else 0.0
    return max(exact, 0.55 * containment + 0.45 * overlap)


def cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


class SemanticScorer:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder
        self._cache: dict[str, tuple[float, ...]] = {}

    def similarity(self, left: str, right: str) -> float:
        lexical = lexical_similarity(left, right)
        if self.embedder is None:
            return lexical
        a, b = self._vectors((left, right))
        semantic = (cosine(a, b) + 1.0) / 2.0
        return max(lexical, 0.82 * semantic + 0.18 * lexical)

    def rank(self, query: str, values: Sequence[str]) -> list[tuple[int, float]]:
        return sorted(
            ((index, self.similarity(query, value)) for index, value in enumerate(values)),
            key=lambda row: (-row[1], row[0]),
        )

    def _vectors(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        missing = [text for text in texts if text not in self._cache]
        if missing:
            values = self.embedder(missing) if self.embedder is not None else []
            if len(values) != len(missing):
                raise ValueError("embedding result count does not match input count")
            for text, vector in zip(missing, values):
                self._cache[text] = tuple(float(value) for value in vector)
        return [self._cache[text] for text in texts]
