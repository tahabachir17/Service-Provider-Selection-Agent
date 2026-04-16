from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCPConnectorConfig:
    """Configuration for future CRM, LinkedIn, or web-search MCP adapters."""

    server_url: str | None
    enable_web_search: bool = False


class MCPProviderEnrichmentClient:
    """Non-operational v1 extension point for provider enrichment tools."""

    def __init__(self, config: MCPConnectorConfig) -> None:
        self.config = config

    def is_configured(self) -> bool:
        return bool(self.config.server_url)
