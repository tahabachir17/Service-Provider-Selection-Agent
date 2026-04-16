from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongTermMemoryConfig:
    """Placeholder for v2 provider history storage."""

    vector_db_path: str
    enabled: bool = False


class ProviderMemoryStore:
    """Optional extension point for Chroma/pgvector-backed provider history."""

    def __init__(self, config: LongTermMemoryConfig) -> None:
        self.config = config

    def is_enabled(self) -> bool:
        return self.config.enabled
