from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import requests


class ModelClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.environ.get("MODEL_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("MODEL_API_KEY", "")
        self.model = model or os.environ.get("MODEL_NAME", "")
        self.timeout = timeout
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("MODEL_BASE_URL, MODEL_API_KEY, and MODEL_NAME are required")

    def complete_text(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"])

    def complete_json(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        enriched = list(messages)
        enriched.append(
            {
                "role": "user",
                "content": "Return one valid JSON object and no surrounding text.",
            }
        )
        text = self.complete_text(
            enriched, temperature=temperature, max_tokens=max_tokens
        )
        return self._parse_object(text)

    @staticmethod
    def _parse_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("model response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("model response must be a JSON object")
        return value


class EmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.environ.get("EMBED_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("EMBED_API_KEY", "")
        self.model = model or os.environ.get("EMBED_MODEL", "")
        self.timeout = timeout
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("EMBED_BASE_URL, EMBED_API_KEY, and EMBED_MODEL are required")

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [[float(value) for value in row["embedding"]] for row in rows]
