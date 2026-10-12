"""Free / low-cost LLM providers that speak the OpenAI-compatible chat API.

Model names change often on free tiers, so the app can also fetch the live list
from each provider ("Fetch available models"). The defaults below are only a
starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    secret_name: str            # name in .streamlit/secrets.toml or environment
    key_url: str                # where to get a free key
    models: tuple[str, ...] = field(default_factory=tuple)
    needs_key: bool = True
    requests_per_minute: int = 20   # polite default for the free tier
    batch_size: int = 15            # reviews per request
    note: str = ""


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        key="groq",
        label="Groq (recommended)",
        base_url="https://api.groq.com/openai/v1",
        secret_name="GROQ_API_KEY",
        key_url="https://console.groq.com/keys",
        models=("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"),
        requests_per_minute=20,
        batch_size=15,
        note="Free account, no card. Very fast. Free plan ≈ 30 requests/min and ~1,000 requests/day per model.",
    ),
    "gemini": Provider(
        key="gemini",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        secret_name="GEMINI_API_KEY",
        key_url="https://aistudio.google.com/apikey",
        models=("gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"),
        requests_per_minute=5,
        batch_size=25,
        note="Free key from Google AI Studio. Daily limits are small, so large batches are used. "
             "Google may use free-tier prompts to improve its products.",
    ),
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter (free models)",
        base_url="https://openrouter.ai/api/v1",
        secret_name="OPENROUTER_API_KEY",
        key_url="https://openrouter.ai/keys",
        models=(),
        requests_per_minute=15,
        batch_size=20,
        note="Pick a model ending in ':free' (press the Models button to list them). Free accounts get a small daily request cap.",
    ),
    "ollama": Provider(
        key="ollama",
        label="Ollama (on your computer)",
        base_url="http://localhost:11434/v1",
        secret_name="OLLAMA_API_KEY",
        key_url="https://ollama.com/download",
        models=("qwen3:8b", "llama3.1:8b", "gemma3:12b"),
        needs_key=False,
        requests_per_minute=0,
        batch_size=8,
        note="100% free and private, no key. Only works when you run the app locally (not on Streamlit Cloud).",
    ),
    "custom": Provider(
        key="custom",
        label="Other compatible API",
        base_url="",
        secret_name="CUSTOM_API_KEY",
        key_url="",
        models=(),
        requests_per_minute=20,
        batch_size=15,
        note="Any service with an OpenAI-compatible /chat/completions endpoint (e.g. Azure / Microsoft Foundry, Together, Mistral).",
    ),
}
