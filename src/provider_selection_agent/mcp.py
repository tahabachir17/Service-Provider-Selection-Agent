from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from provider_selection_agent.config import Settings
from provider_selection_agent.models import ProviderProfile

MCPRequester = Callable[[str, dict[str, Any]], dict[str, Any]]
ALLOWED_ENRICHMENT_FIELDS = {
    "price",
    "currency",
    "expertise",
    "location",
    "availability",
    "portfolio_summary",
    "references",
    "notes",
}


@dataclass(frozen=True)
class MCPConnectorConfig:
    """Configuration for HTTP MCP bridge adapters used for provider enrichment."""

    server_url: str | None
    enable_web_search: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1"
    llm_base_url: str | None = None
    enrich_fields: tuple[str, ...] = ()
    timeout_seconds: int = 20

    @classmethod
    def from_settings(cls, settings: Settings) -> MCPConnectorConfig:
        return cls(
            server_url=settings.mcp_server_url,
            enable_web_search=settings.enable_web_search,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            llm_base_url=settings.llm_base_url,
            enrich_fields=settings.mcp_enrich_fields,
            timeout_seconds=settings.mcp_timeout_seconds,
        )


class MCPProviderEnrichmentClient:
    """HTTP client for an MCP gateway that enriches provider records with evidence."""

    def __init__(
        self, config: MCPConnectorConfig, requester: MCPRequester | None = None
    ) -> None:
        self.config = config
        self._requester = requester or _post_json

    def is_configured(self) -> bool:
        return bool(self.config.server_url)

    def enrich_providers(
        self, providers: list[ProviderProfile]
    ) -> tuple[list[ProviderProfile], dict[str, Any]]:
        if not self.is_configured():
            return providers, {
                "attempted": False,
                "reason": "MCP_SERVER_URL is not configured",
                "applied_updates": [],
            }

        wanted_fields = self._wanted_fields(providers)
        if not wanted_fields:
            return providers, {
                "attempted": False,
                "reason": "No enrichable fields were requested",
                "applied_updates": [],
            }

        payload = {
            "providers": [provider.model_dump(mode="json") for provider in providers],
            "wanted_fields": wanted_fields,
            "web_search_enabled": self.config.enable_web_search,
            "_timeout_seconds": self.config.timeout_seconds,
            "llm": {
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "base_url": self.config.llm_base_url,
            },
        }
        response = self._requester(self.config.server_url or "", payload)
        return _apply_enrichment_payload(providers, response, wanted_fields)

    def _wanted_fields(self, providers: list[ProviderProfile]) -> list[str]:
        configured = [
            field for field in self.config.enrich_fields if field in ALLOWED_ENRICHMENT_FIELDS
        ]
        if configured:
            return configured

        inferred: list[str] = []
        if any(provider.price is None for provider in providers):
            inferred.append("price")
        if any(provider.expertise == "unknown" for provider in providers):
            inferred.append("expertise")
        if any(provider.location == "unknown" for provider in providers):
            inferred.append("location")
        if any(provider.availability is None for provider in providers):
            inferred.append("availability")
        return inferred


class MCPProviderDiscoveryClient:
    """HTTP client for provider discovery via an MCP-backed sourcing bridge."""

    def __init__(
        self, config: MCPConnectorConfig, requester: MCPRequester | None = None
    ) -> None:
        self.config = config
        self._requester = requester or _post_json

    def is_configured(self) -> bool:
        return bool(self.config.server_url)

    def discover_providers(
        self,
        *,
        project_context: str,
        target_fields: list[str],
        preferred_location: str,
        remote_ok: bool,
        max_results: int,
    ) -> tuple[list[ProviderProfile], dict[str, Any]]:
        if not self.is_configured():
            raise RuntimeError("MCP_SERVER_URL is not configured")

        payload = {
            "operation": "discover_providers",
            "project_context": project_context,
            "target_fields": target_fields,
            "preferred_location": preferred_location,
            "remote_ok": remote_ok,
            "max_results": max_results,
            "_timeout_seconds": self.config.timeout_seconds,
            "strict_rules": {
                "zero_hallucination": True,
                "no_directories": True,
                "limit": max_results,
            },
            "web_search_enabled": self.config.enable_web_search,
            "llm": {
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "base_url": self.config.llm_base_url,
            },
        }
        response = self._requester(self.config.server_url or "", payload)
        return _normalize_discovery_response(response)


def _apply_enrichment_payload(
    providers: list[ProviderProfile], response: dict[str, Any], wanted_fields: list[str]
) -> tuple[list[ProviderProfile], dict[str, Any]]:
    updates_by_name: dict[str, dict[str, Any]] = {}
    evidence_by_name: dict[str, list[dict[str, Any]]] = {}

    for item in response.get("providers", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        raw_fields = item.get("fields", {})
        if not isinstance(raw_fields, dict):
            raw_fields = {}
        filtered_fields = {
            key: value
            for key, value in raw_fields.items()
            if key in ALLOWED_ENRICHMENT_FIELDS
            and key in wanted_fields
            and value not in (None, "", "unknown")
        }
        if filtered_fields:
            updates_by_name[name.lower()] = filtered_fields
        raw_evidence = item.get("evidence", [])
        if isinstance(raw_evidence, list):
            evidence_by_name[name.lower()] = [
                evidence for evidence in raw_evidence if isinstance(evidence, dict)
            ]

    updated_providers: list[ProviderProfile] = []
    applied_updates: list[dict[str, Any]] = []
    for provider in providers:
        update_fields = updates_by_name.get(provider.name.lower(), {})
        if not update_fields:
            updated_providers.append(provider)
            continue
        try:
            updated_provider = ProviderProfile.model_validate(
                {
                    **provider.model_dump(mode="json"),
                    **update_fields,
                }
            )
        except ValidationError:
            updated_providers.append(provider)
            continue
        updated_providers.append(updated_provider)
        applied_updates.append(
            {
                "provider_name": provider.name,
                "applied_fields": sorted(update_fields),
                "evidence": evidence_by_name.get(provider.name.lower(), []),
            }
        )

    return updated_providers, {
        "attempted": True,
        "wanted_fields": wanted_fields,
        "applied_updates": applied_updates,
        "updated_provider_count": len(applied_updates),
    }


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    timeout_seconds = int(payload.pop("_timeout_seconds", 300))
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover - network failure path
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MCP request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"MCP request failed: {exc.reason}") from exc


def _normalize_discovery_response(
    response: dict[str, Any],
) -> tuple[list[ProviderProfile], dict[str, Any]]:
    raw_providers = response.get("providers", [])
    if not isinstance(raw_providers, list):
        raise RuntimeError("MCP discovery response must include a 'providers' list")

    profiles: list[ProviderProfile] = []
    for item in raw_providers:
        if not isinstance(item, dict):
            continue
        normalized = {
            "name": item.get("name", "unknown"),
            "type": item.get("type", "unknown"),
            "expertise": item.get("expertise", "unknown"),
            "location": item.get("location", "unknown"),
            "price": item.get("price", None),
            "currency": item.get("currency", "unknown"),
            "portfolio_summary": item.get("portfolio_summary", "unknown"),
            "references": _format_references(item.get("evidence", [])),
            "notes": _format_notes(item),
        }
        try:
            profiles.append(ProviderProfile.model_validate(normalized))
        except ValidationError:
            continue

    audit = {
        "attempted": True,
        "provider_count": len(profiles),
        "raw_provider_count": len(raw_providers),
        "search_summary": response.get("search_summary", ""),
    }
    return profiles, audit


def _format_references(evidence: Any) -> str:
    if not isinstance(evidence, list):
        return "unknown"
    refs: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        field_name = str(item.get("field", "")).strip()
        if source and field_name:
            refs.append(f"{field_name}: {source}")
        elif source:
            refs.append(source)
    return " | ".join(refs) if refs else "unknown"


def _format_notes(item: dict[str, Any]) -> str:
    parts: list[str] = []
    source = str(item.get("source_type", "")).strip()
    if source:
        parts.append(f"source_type={source}")
    rationale = str(item.get("relevance_rationale", "")).strip()
    if rationale:
        parts.append(rationale)
    return " | ".join(parts) if parts else "unknown"
