from __future__ import annotations

import json

from provider_selection_agent.config import Settings
from provider_selection_agent.mcp import MCPConnectorConfig, MCPProviderDiscoveryClient
from provider_selection_agent.sourcing import write_discovery_output


def _settings() -> Settings:
    return Settings(
        llm_provider="gemini",
        llm_api_key="test-key",
        llm_model="gemini-2.5-flash",
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        log_level="INFO",
        vector_db_path=".local/vector_store",
        enable_web_search=True,
        mcp_server_url="https://example.invalid/mcp",
        mcp_enrich_fields=("price", "expertise", "location", "availability"),
        mcp_timeout_seconds=120,
    )


def test_discovery_client_normalizes_provider_results() -> None:
    client = MCPProviderDiscoveryClient(
        MCPConnectorConfig.from_settings(_settings()),
        requester=lambda _url, _payload: {
            "providers": [
                {
                    "name": "Example Labs",
                    "type": "Agency",
                    "expertise": ["Full-Stack Web Development", "Data Architecture"],
                    "location": "Warsaw, Poland",
                    "price": 120.0,
                    "currency": "USD",
                    "portfolio_summary": "Built marketplaces. Built search systems.",
                    "evidence": [
                        {"field": "location", "source": "https://example.com/about"},
                        {"field": "price", "source": "https://example.com/pricing"},
                    ],
                    "source_type": "provider_site",
                    "relevance_rationale": "Strong fit for marketplace backend and search.",
                }
            ],
            "search_summary": "1 relevant provider found",
        },
    )

    profiles, audit = client.discover_providers(
        project_context="Need marketplace backend and search",
        target_fields=["Full-Stack Web Development", "Data Architecture"],
        preferred_location="EMEA",
        remote_ok=True,
        max_results=5,
    )

    assert len(profiles) == 1
    assert profiles[0].name == "Example Labs"
    assert profiles[0].expertise == "Full-Stack Web Development, Data Architecture"
    assert profiles[0].references == "location: https://example.com/about | price: https://example.com/pricing"
    assert "provider_site" in profiles[0].notes
    assert audit["provider_count"] == 1


def test_write_discovery_output_writes_project_compatible_json(tmp_path) -> None:
    profiles = [
        MCPProviderDiscoveryClient(
            MCPConnectorConfig.from_settings(_settings()),
            requester=lambda _url, _payload: {
                "providers": [
                    {
                        "name": "Example Labs",
                        "type": "Agency",
                        "expertise": ["Full-Stack Web Development", "Data Architecture"],
                        "location": "Warsaw, Poland",
                        "price": 120.0,
                        "currency": "USD",
                        "portfolio_summary": "Built marketplaces. Built search systems.",
                    }
                ]
            },
        ).discover_providers(
            project_context="EdTech marketplace for tutoring and courses",
            target_fields=["Full-Stack Web Development", "Data Architecture"],
            preferred_location="EMEA",
            remote_ok=True,
            max_results=3,
        )[0][0]
    ]
    output_path = write_discovery_output(
        output_path=tmp_path / "providers.json",
        profiles=profiles,
        audit={"provider_count": 1},
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["providers"][0]["name"] == "Example Labs"
    assert payload["discovery_audit"]["provider_count"] == 1
