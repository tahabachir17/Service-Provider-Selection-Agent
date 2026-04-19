from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from provider_selection_agent.config import Settings, load_settings
from provider_selection_agent.mcp import MCPConnectorConfig, MCPProviderDiscoveryClient
from provider_selection_agent.models import ProviderProfile


def discover_providers_via_mcp(
    *,
    project_context: str,
    target_fields: list[str],
    preferred_location: str,
    remote_ok: bool,
    max_results: int,
    settings: Settings,
) -> tuple[list[ProviderProfile], dict[str, Any]]:
    client = MCPProviderDiscoveryClient(MCPConnectorConfig.from_settings(settings))
    return client.discover_providers(
        project_context=project_context,
        target_fields=target_fields,
        preferred_location=preferred_location,
        remote_ok=remote_ok,
        max_results=max_results,
    )


def execute_sourcing_run(
    context: str,
    fields: str,
    location: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Run provider discovery and return normalized provider payloads plus audit data."""

    settings = load_settings()
    target_fields = [item.strip() for item in fields.split(",") if item.strip()]
    config = replace(
        MCPConnectorConfig.from_settings(settings),
        server_url=settings.mcp_server_url or "http://local-bridge/enrich",
    )
    client = MCPProviderDiscoveryClient(
        config,
        requester=_local_bridge_request,
    )
    profiles, audit = client.discover_providers(
        project_context=context,
        target_fields=target_fields,
        preferred_location=location,
        remote_ok=True,
        max_results=max_results,
    )
    return {
        "providers": [_profile_to_discovered_provider(profile) for profile in profiles],
        "search_summary": audit.get("search_summary", ""),
        "audit": audit,
    }


def write_discovery_output(
    *,
    output_path: Path,
    profiles: list[ProviderProfile],
    audit: dict[str, Any],
) -> Path:
    payload = {
        "providers": [profile.model_dump(mode="json") for profile in profiles],
        "discovery_audit": audit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _local_bridge_request(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    from provider_selection_agent.mcp_bridge import handle_legacy_discovery

    return handle_legacy_discovery(payload, load_settings())


def _profile_to_discovered_provider(profile: ProviderProfile) -> dict[str, Any]:
    evidence = []
    if profile.references and profile.references != "unknown":
        for item in profile.references.split(" | "):
            raw = item.strip()
            if not raw:
                continue
            if ": " in raw:
                field_name, source = raw.split(": ", 1)
            else:
                field_name, source = "source", raw
            evidence.append({"field": field_name, "source": source})

    expertise = (
        [item.strip() for item in profile.expertise.split(",") if item.strip()]
        if profile.expertise and profile.expertise != "unknown"
        else []
    )

    return {
        "name": profile.name,
        "type": profile.type,
        "expertise": expertise,
        "location": profile.location,
        "price": profile.price,
        "currency": profile.currency,
        "portfolio_summary": profile.portfolio_summary,
        "source_type": "provider_site",
        "relevance_rationale": profile.notes if profile.notes != "unknown" else "",
        "evidence": evidence,
    }
