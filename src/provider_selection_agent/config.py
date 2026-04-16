from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    log_level: str
    vector_db_path: str
    enable_web_search: bool
    mcp_server_url: str | None


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", ".local/vector_store"),
        enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true",
        mcp_server_url=os.getenv("MCP_SERVER_URL") or None,
    )
