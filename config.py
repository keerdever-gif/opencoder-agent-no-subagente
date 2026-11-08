"""Environment-based configuration for OpenCoder-Agent."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'")
    return value or default


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer (received {raw!r}).") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum} (received {value}).")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number (received {raw!r}).") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum} (received {value}).")
    return value


@dataclass(frozen=True)
class AISlot:
    index: int
    key: str
    model: str
    base_url: str = ""


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    ai_slots: list[AISlot]
    ai_temperature: float
    ai_max_tokens: int
    ai_max_tool_rounds: int
    ai_timeouts_seconds: int
    verify_code: bool
    max_fix_attempts: int
    code_run_timeout: int
    bot_name: str
    memory_limit: int
    memory_messages: int
    workspace: str
    port: int
    web_access_token: str | None
    web_host: str
    github_token: str | None
    vercel_token: str | None
    flux_api_key_1: str | None
    flux_api_key_2: str | None
    flux_base_url: str
    flux_endpoint: str

    @property
    def ai_model(self) -> str:
        return self.ai_slots[0].model if self.ai_slots else "unknown"


_DEFAULT_MODEL = _env("AI_MODEL", "gemini-flash-lite-latest") or "gemini-flash-lite-latest"


def _load_slots(default_model: str) -> list[AISlot]:
    slots: list[AISlot] = []
    main_key = _env("AI_API_KEY")
    if main_key:
        slots.append(AISlot(0, main_key, _env("AI_MODEL", default_model) or default_model, _env("AI_API_BASE_URL", "") or ""))

    for index in range(1, 101):
        key = _env(f"AI_API_KEY_{index}")
        if not key:
            continue
        slots.append(AISlot(index, key, _env(f"AI_MODEL_{index}", default_model) or default_model, _env(f"AI_API_BASE_URL_{index}", "") or ""))
    return sorted(slots, key=lambda slot: slot.index)


def load_settings() -> Settings:

    return Settings(
        database_url=_env("DATABASE_URL"),
        ai_slots=_load_slots(_DEFAULT_MODEL),
        ai_temperature=_float("AI_TEMPERATURE", 0.7, 0.0, 2.0),
        ai_max_tokens=_int("AI_MAX_TOKENS", 4096, 16, 32768),
        ai_max_tool_rounds=_int("MAX_SAFETY_ROUNDS", 100, 1, 500),
        ai_timeouts_seconds=_int("AI_TIMEOUT_SECONDS", 600, 10, 3600),
        verify_code=(_env("VERIFY_CODE", "1") or "1").lower() not in {"0", "false", "no", "off"},
        max_fix_attempts=_int("MAX_FIX_ATTEMPTS", 8, 1, 30),
        code_run_timeout=_int("CODE_RUN_TIMEOUT", 20, 5, 120),
        bot_name=_env("BOT_NAME", "OpenCoder-Agent") or "OpenCoder-Agent",
        memory_limit=_int("MEMORY_LIMIT", 120, 0, 10080),
        memory_messages=_int("MEMORY_MESSAGES", 12, 2, 200),
        workspace=_env("OPENCODER_WORKSPACE", _env("SUBTOM_WORKSPACE", "./workspace")) or "./workspace",
        port=_int("PORT", 8080, 1, 65535),
        web_access_token=_env("WEB_ACCESS_TOKEN"),
        web_host=_env("WEB_HOST", "0.0.0.0") or "0.0.0.0",
        github_token=_env("GITHUB_TOKEN"),
        vercel_token=_env("VERCEL_TOKEN"),
        flux_api_key_1=_env("FLUX_API_KEY_1"),
        flux_api_key_2=_env("FLUX_API_KEY_2"),
        flux_base_url=_env("FLUX_BASE_URL", "https://api.bfl.ai/v1") or "https://api.bfl.ai/v1",
        flux_endpoint=_env("FLUX_ENDPOINT", "flux-2-flex") or "flux-2-flex",
    )


config = load_settings()
