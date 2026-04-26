from __future__ import annotations

from typing import Any

import requests


class VLLMSummaryGenerator:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "demo-key",
        model: str = "summary-sft",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, prompt: str, context: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()