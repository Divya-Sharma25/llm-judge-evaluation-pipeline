"""
Thin wrapper around Groq's OpenAI-compatible chat completions endpoint.
Tracks token usage and call count so the pipeline can report judge cost (requirement #4).
"""
import requests
from . import config


class GroqClient:
    def __init__(self):
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, model: str, messages: list, temperature: float = 0.0, max_tokens: int = 1024) -> dict:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GROQ_API_KEY in your shell."
            )
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(config.GROQ_BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        self.total_calls += 1
        usage = data.get("usage", {})
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)

        text = data["choices"][0]["message"]["content"]
        return {"text": text, "usage": usage, "raw": data}

    def usage_summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
