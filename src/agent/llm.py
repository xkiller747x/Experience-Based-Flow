"""LLM gateway supporting both OpenAI-compatible API and local llama.cpp server."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import requests

from src.agent.config import load_llm_config


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    raw: str
    provider: Literal["openai", "local", "douban"]


class LLMGateway:
    """
    Unified LLM interface supporting:
    - OpenAI-compatible APIs (provider="openai") — works with OpenAI, DeepSeek, etc.
    - Local llama.cpp server (provider="local", localhost:8080)
    - Douban/Volcengine (provider="douban", ark.cn-beijing.volces.com)

    Args:
        provider: "openai", "local", or "douban". If None, read from local config.
        api_key: API key. If None, read from local config.
        local_url: URL for local llama.cpp server. If None, read from local config.
        model: Model name. If None, read from local config.
        timeout: Request timeout in seconds. If None, read from local config.
    """

    def __init__(
        self,
        provider: Literal["openai", "local", "douban"] | None = None,
        api_key: str | None = None,
        local_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        config = load_llm_config()
        self.provider = provider or config.provider
        self._api_key = api_key if api_key is not None else config.api_key
        self._local_url = (local_url or config.local_url).rstrip("/")
        self._model = model or config.model
        self._timeout = timeout if timeout is not None else config.timeout
        self._base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")

        if self.provider in {"openai", "douban"} and not self._api_key:
            raise ValueError(
                f"{self.provider} api_key is not set. Add it to config/llm.local.json "
                "or pass api_key explicitly."
            )
        if self.provider not in {"openai", "local", "douban"}:
            raise ValueError(f"provider must be 'openai', 'local', or 'douban', got '{provider}'")

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int | None = None) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt / task description.
            system_prompt: Optional system-level instruction.
            max_tokens: Optional max completion tokens.

        Returns:
            The LLM's response text.

        Raises:
            LLMError: If the request fails or the response is malformed.
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, max_tokens=max_tokens)
        elif self.provider == "douban":
            return self._generate_douban(prompt, system_prompt, max_tokens=max_tokens)
        else:
            return self._generate_local(prompt, system_prompt)

    # ------------------------------------------------------------------
    # OpenAI-compatible API (OpenAI, DeepSeek, etc.)
    # ------------------------------------------------------------------
    def _generate_openai(self, prompt: str, system_prompt: str, *, max_tokens: int | None = None) -> str:
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
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Try requests first, fall back to curl for SSL compatibility
        try:
            return self._generate_openai_requests(url, headers, payload)
        except requests.exceptions.SSLError:
            return self._generate_openai_curl(url, headers, payload)

    def _generate_openai_requests(self, url: str, headers: dict, payload: dict) -> str:
        """Try requests library (may fail with SSL on some servers)."""
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            body = resp.text
        except requests.exceptions.HTTPError as exc:
            raise LLMError(
                f"OpenAI API HTTP {resp.status_code}: {resp.text[:300]}",
                provider="openai",
                status_code=resp.status_code,
            ) from exc
        except requests.exceptions.SSLError:
            raise  # let parent handle SSL fallback
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"OpenAI API connection failed: {exc}",
                provider="openai",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                f"OpenAI API timed out after {self._timeout}s",
                provider="openai",
            ) from exc
        return self._parse_openai_response(body)

    def _generate_openai_curl(self, url: str, headers: dict, payload: dict) -> str:
        """Fallback: use curl.exe (Schannel) when requests/OpenSSL fails."""
        # Write payload to a temp file (avoids shell escaping hell on Windows)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        try:
            json.dump(payload, tmp, ensure_ascii=False)
            tmp.close()

            auth = headers.get("Authorization", "")
            cmd = [
                "curl.exe", "-s", "-m", str(self._timeout or 30),
                url,
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: {auth}",
                "-d", f"@{tmp.name}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout or 30)
            if result.returncode != 0:
                raise LLMError(
                    f"curl failed (exit {result.returncode}): {result.stderr[:500]}",
                    provider="openai",
                )
            return self._parse_openai_response(result.stdout)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def _parse_openai_response(self, body: str) -> str:
        """Parse OpenAI-compatible JSON response body."""
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
            content = choices[0]["message"]["content"]
            if not content:
                content = choices[0]["message"].get("reasoning_content", "")
            return content.strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"OpenAI API response is missing expected fields: {parsed}",
                provider="openai",
            ) from exc

    # ------------------------------------------------------------------
    # Douban/Volcengine API
    # ------------------------------------------------------------------
    def _generate_douban(self, prompt: str, system_prompt: str, *, max_tokens: int | None = None) -> str:
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
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            body = resp.text
        except requests.exceptions.HTTPError as exc:
            raise LLMError(
                f"Douban API HTTP {resp.status_code}: {resp.text[:300]}",
                provider="douban",
                status_code=resp.status_code,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Douban API connection failed: {exc}",
                provider="douban",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                f"Douban API timed out after {self._timeout}s",
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
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            body = resp.text
        except requests.exceptions.HTTPError as exc:
            raise LLMError(
                f"Local LLM HTTP {resp.status_code}: {resp.text[:300]}",
                provider="local",
                status_code=resp.status_code,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(
                f"Local LLM connection failed (is llama.cpp server running at "
                f"{self._local_url}?): {exc}",
                provider="local",
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(
                f"Local LLM timed out after {self._timeout}s",
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
