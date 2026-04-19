from __future__ import annotations

from provider_selection_agent.mcp import MCPConnectorConfig, MCPProviderEnrichmentClient
from provider_selection_agent.models import ProviderProfile


def test_mcp_enrichment_applies_valid_updates() -> None:
    provider = ProviderProfile(name="Alpha", price=None, expertise="unknown", location="unknown")
    client = MCPProviderEnrichmentClient(
        MCPConnectorConfig(
            server_url="https://example.invalid/mcp",
            enable_web_search=True,
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
            enrich_fields=("price", "expertise", "location"),
            timeout_seconds=120,
        ),
        requester=lambda _url, _payload: {
            "providers": [
                {
                    "name": "Alpha",
                    "fields": {
                        "price": 1200,
                        "expertise": "MLOps and LLM systems",
                        "location": "Remote",
                    },
                    "evidence": [
                        {
                            "field": "price",
                            "source": "https://example.com/alpha",
                            "summary": "Public pricing page",
                        }
                    ],
                }
            ]
        },
    )

    updated_providers, audit = client.enrich_providers([provider])

    assert updated_providers[0].price == 1200
    assert updated_providers[0].expertise == "MLOps and LLM systems"
    assert updated_providers[0].location == "Remote"
    assert audit["updated_provider_count"] == 1
    assert audit["applied_updates"][0]["provider_name"] == "Alpha"


def test_mcp_enrichment_is_skipped_without_server_url() -> None:
    provider = ProviderProfile(name="Alpha")
    client = MCPProviderEnrichmentClient(MCPConnectorConfig(server_url=None))

    updated_providers, audit = client.enrich_providers([provider])

    assert updated_providers == [provider]
    assert audit["attempted"] is False
