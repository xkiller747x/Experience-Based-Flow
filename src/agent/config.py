"""Local configuration loader for LLM provider settings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_ENV_VAR = "LOGISTIC_AI_LLM_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "llm.local.json"
API_KEY_PLACEHOLDERS = {"your-api-key-here", "your-key", "changeme", "change-me"}


class LLMConfigError(ValueError):
    """Raised when the local LLM config file is invalid."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "douban"
    api_key: str | None = None
    model: str = "doubao-seed-1-8-251228"
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    local_url: str = "http://localhost:8080"
    timeout: int = 120


def get_llm_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured local LLM config path."""

    if path is not None:
        return Path(path)

    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path)

    return DEFAULT_CONFIG_PATH


def load_llm_config(path: str | os.PathLike[str] | None = None) -> LLMConfig:
    """Load LLM settings from a local JSON file, falling back to safe defaults."""

    config_path = get_llm_config_path(path)
    if not config_path.exists():
        return LLMConfig()

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LLMConfigError(f"Invalid JSON in LLM config file: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise LLMConfigError(f"LLM config file must contain a JSON object: {config_path}")

    return _parse_config(raw_config, config_path)


def _parse_config(raw_config: dict[str, Any], config_path: Path) -> LLMConfig:
    config = LLMConfig()

    provider = _get_str(raw_config, "provider", config.provider, config_path).lower()
    if provider not in {"openai", "local", "douban"}:
        raise LLMConfigError(
            f"LLM config provider must be 'openai', 'local', or 'douban': {config_path}"
        )

    return LLMConfig(
        provider=provider,
        api_key=_get_optional_str(raw_config, "api_key", config_path),
        model=_get_str(raw_config, "model", config.model, config_path),
        base_url=_get_str(raw_config, "base_url", config.base_url, config_path).rstrip("/"),
        local_url=_get_str(raw_config, "local_url", config.local_url, config_path).rstrip("/"),
        timeout=_get_int(raw_config, "timeout", config.timeout, config_path),
    )


def _get_str(raw_config: dict[str, Any], key: str, default: str, config_path: Path) -> str:
    value = raw_config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigError(f"LLM config field '{key}' must be a non-empty string: {config_path}")
    return value.strip()


def _get_optional_str(raw_config: dict[str, Any], key: str, config_path: Path) -> str | None:
    value = raw_config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LLMConfigError(f"LLM config field '{key}' must be a string: {config_path}")
    value = value.strip()
    if value.lower() in API_KEY_PLACEHOLDERS:
        return None
    return value or None


def _get_int(raw_config: dict[str, Any], key: str, default: int, config_path: Path) -> int:
    value = raw_config.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise LLMConfigError(f"LLM config field '{key}' must be a positive integer: {config_path}")
    return value
