"""LLM gateway supporting both Doubao API and local llama.cpp server."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Literal


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    raw: str
    provider: Literal["openai", "local"]


class LLMGateway:
    """
    Unified LLM interface supporting:
    - OpenAI GPT (provider="openai")
    - Local llama.cpp server (provider="local", localhost:8080)
    - Douban/Volcengine (provider="douban", ark.cn-beijing.volces.com)

    Args:
        provider: "openai", "local", or "douban". Defaults to "douban".
        api_key: API key. If None, read from environment variable.
        local_url: URL for local llama.cpp server. Defaults to "http://localhost:8080".
        model: Model name. Defaults to "doubao-seed-1-8".
        timeout: Request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        provider: Literal["openai", "local", "douban"] = "douban",
        api_key: str | None = None,
        local_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.provider = provider
        self._api_key = api_key or os.environ.get("DOUBAN_API_KEY", "5f0d1c05-af99-4fb7-939d-e6529a31b04e")
        self._local_url = (local_url or "http://localhost:8080").rstrip("/")
        self._model = model or "doubao-seed-1-8-251228"
        self._timeout = timeout if timeout is not None else 120
        self._base_url = os.environ.get("DOUBAN_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

        if self.provider == "openai" and not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set and no api_key was provided. "
                "Set the environment variable or pass api_key explicitly."
            )
        if self.provider == "douban" and not self._api_key:
            raise ValueError(
                "DOUBAN_API_KEY environment variable is not set and no api_key was provided. "
                "Set the environment variable or pass api_key explicitly."
            )
        if self.provider not in {"openai", "local", "douban"}:
            raise ValueError(f"provider must be 'openai', 'local', or 'douban', got '{provider}'")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt / task description.
            system_prompt: Optional system-level instruction.

        Returns:
            The LLM's response text.

        Raises:
            LLMError: If the request fails or the response is malformed.
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "douban":
            return self._generate_douban(prompt, system_prompt)
        else:
            return self._generate_local(prompt, system_prompt)

    # ------------------------------------------------------------------
    # OpenAI API
    # ------------------------------------------------------------------
    def _generate_openai(self, prompt: str, system_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LLMError(
                f"OpenAI API HTTP {exc.code}: {exc.reason}",
                provider="openai",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"OpenAI API connection failed: {exc.reason}",
                provider="openai",
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"OpenAI API returned non-JSON response: {body[:500]}",
                provider="openai",
            ) from exc

        try:
            choices = parsed["choices"]
            if not choices:
                raise LLMError("OpenAI API response has no choices", provider="openai")
            return choices[0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"OpenAI API response is missing expected fields: {parsed}",
                provider="openai",
            ) from exc

    # ------------------------------------------------------------------
    # Douban/Volcengine API
    # ------------------------------------------------------------------
    def _generate_douban(self, prompt: str, system_prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LLMError(
                f"Douban API HTTP {exc.code}: {exc.reason}",
                provider="douban",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Douban API connection failed: {exc.reason}",
                provider="douban",
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Douban API returned non-JSON response: {body[:500]}",
                provider="douban",
            ) from exc

        try:
            choices = parsed["choices"]
            if not choices:
                raise LLMError("Douban API response has no choices", provider="douban")
            return choices[0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Douban API response is missing expected fields: {parsed}",
                provider="douban",
            ) from exc

    # ------------------------------------------------------------------
    # Local llama.cpp server
    # ------------------------------------------------------------------
    def _generate_local(self, prompt: str, system_prompt: str) -> str:
        """Call a llama.cpp server via its completion API (v1/completions)."""

        url = f"{self._local_url}/v1/completions"
        headers = {
            "Content-Type": "application/json",
        }

        full_prompt = (system_prompt + "\n\n" + prompt).strip() if system_prompt else prompt

        payload = {
            "prompt": full_prompt,
            "stream": False,
            "temperature": 0.0,
            "n_predict": 2048,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LLMError(
                f"Local LLM HTTP {exc.code}: {exc.reason}",
                provider="local",
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Local LLM connection failed (is llama.cpp server running at "
                f"{self._local_url}?): {exc.reason}",
                provider="local",
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Local LLM returned non-JSON response: {body[:500]}",
                provider="local",
            ) from exc

        try:
            content: str = parsed["choices"][0]["text"]
            return content.strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Local LLM response missing expected fields: {parsed}",
                provider="local",
            ) from exc


class LLMError(Exception):
    """Raised when LLM generation fails."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
