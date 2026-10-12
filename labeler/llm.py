"""Thin wrapper around the OpenAI SDK that works with any OpenAI-compatible provider."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import openai
from openai import OpenAI


class QuotaError(RuntimeError):
    """Rate/daily limit reached even after retries — the run should pause."""


class AuthError(RuntimeError):
    """The API key is missing or rejected."""


class ModelError(RuntimeError):
    """Anything else that went wrong for one request."""


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Send a system + user prompt, get back a parsed JSON object."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        temperature: float = 0.0,
        requests_per_minute: int = 0,
        timeout: float = 120.0,
        max_retries: int = 4,
        sdk_client: object | None = None,
        sleep_fn=time.sleep,
    ):
        if not model:
            raise ValueError("Choose a model first.")
        self.model = model
        self.temperature = temperature
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.usage = Usage()
        self._last_call = 0.0
        self._sleep = sleep_fn
        self._json_mode = True
        self.client = sdk_client or OpenAI(
            base_url=base_url or None,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=max_retries,   # SDK retries 429/5xx with exponential back-off
        )

    # ── public ──────────────────────────────────────────────────────────────
    def chat_json(self, system: str, user: str) -> dict:
        self._throttle()
        try:
            resp = self._create(system, user)
        except openai.AuthenticationError as exc:
            raise AuthError("The API key was rejected. Check it and try again.") from exc
        except openai.PermissionDeniedError as exc:
            raise AuthError(f"Access denied for this model: {_msg(exc)}") from exc
        except openai.RateLimitError as exc:
            raise QuotaError(f"Rate limit or daily quota reached: {_msg(exc)}") from exc
        except openai.NotFoundError as exc:
            raise ModelError(f"Model '{self.model}' was not found at this provider.") from exc
        except openai.APIConnectionError as exc:
            raise ModelError(f"Could not connect to the provider: {_msg(exc)}") from exc
        except openai.APIError as exc:
            raise ModelError(_msg(exc)) from exc

        self.usage.requests += 1
        if getattr(resp, "usage", None):
            self.usage.prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
            self.usage.completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0

        text = (resp.choices[0].message.content or "").strip()
        return parse_json_object(text)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self.client.models.list())
        except openai.AuthenticationError as exc:
            raise AuthError("The API key was rejected.") from exc
        except openai.APIError as exc:
            raise ModelError(_msg(exc)) from exc

    # ── internals ───────────────────────────────────────────────────────────
    def _create(self, system: str, user: str):
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if self._json_mode:
            try:
                return self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            except openai.BadRequestError as exc:
                # Some models/providers don't support JSON mode — fall back to plain text.
                if "response_format" in str(exc).lower() or "json" in str(exc).lower():
                    self._json_mode = False
                else:
                    raise
        return self.client.chat.completions.create(**kwargs)

    def _throttle(self) -> None:
        if not self.min_interval:
            return
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and <think> blocks."""
    cleaned = _FENCE.sub("", _THINK.sub("", text).strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ModelError(f"The model did not return JSON: {text[:160]!r}")
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ModelError(f"The model returned invalid JSON: {text[:160]!r}") from exc
    if isinstance(data, list):
        data = {"results": data}
    if not isinstance(data, dict):
        raise ModelError("The model returned JSON that is not an object.")
    return data


def _msg(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
    return str(exc)[:300]
