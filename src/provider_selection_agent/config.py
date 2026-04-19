from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MCP_ENRICH_FIELDS = ("price", "expertise", "location", "availability")


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    log_level: str
    vector_db_path: str
    enable_web_search: bool
    mcp_server_url: str | None
    mcp_enrich_fields: tuple[str, ...]
    mcp_timeout_seconds: int

    @property
    def openai_api_key(self) -> str | None:
        return self.llm_api_key

    @property
    def openai_model(self) -> str:
        return self.llm_model

    @property
    def api_key_env_hint(self) -> str:
        if self.llm_provider == "gemini":
            return "GEMINI_API_KEY or LLM_API_KEY"
        if self.llm_provider == "groq":
            return "GROQ_API_KEY or LLM_API_KEY"
        return "OPENAI_API_KEY or LLM_API_KEY"


def load_settings(*, load_env_file: bool = True) -> Settings:
    if load_env_file:
        load_dotenv()
    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not llm_provider:
        if os.getenv("GROQ_API_KEY"):
            llm_provider = "groq"
        elif os.getenv("GEMINI_API_KEY"):
            llm_provider = "gemini"
        else:
            llm_provider = "openai"

    if llm_provider not in {"openai", "gemini", "groq"}:
        raise ValueError("LLM_PROVIDER must be one of 'openai', 'gemini', or 'groq'")

    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None
    if llm_provider == "gemini":
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or None
        llm_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gemini-2.5-flash"
        llm_base_url = (
            os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or GEMINI_OPENAI_BASE_URL
        )
    elif llm_provider == "groq":
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or None
        llm_model = os.getenv("LLM_MODEL") or "openai/gpt-oss-20b"
        llm_base_url = (
            os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or GROQ_OPENAI_BASE_URL
        )
    else:
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
        llm_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1"
        llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None

    return Settings(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", ".local/vector_store"),
        enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true",
        mcp_server_url=os.getenv("MCP_SERVER_URL") or None,
        mcp_enrich_fields=_parse_csv_env("MCP_ENRICH_FIELDS", DEFAULT_MCP_ENRICH_FIELDS),
        mcp_timeout_seconds=int(os.getenv("MCP_TIMEOUT_SECONDS", "300")),
    )


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    fields = tuple(
        dict.fromkeys(item.strip() for item in raw.split(",") if item.strip())
    )
    return fields or default
