from __future__ import annotations

from pathlib import Path

import pytest

from provider_selection_agent.loaders import load_criteria, load_providers
from provider_selection_agent.models import CriteriaConfig, ProviderProfile
from provider_selection_agent.scoring import score_providers

ROOT = Path(__file__).resolve().parents[1]


def test_loads_example_providers_and_criteria() -> None:
    providers = load_providers(ROOT / "examples" / "providers.csv")
    criteria = load_criteria(ROOT / "config" / "criteria.yaml")

    assert len(providers) == 4
    assert criteria.normalized_weights == {
        "price": 0.3,
        "expertise": 0.35,
        "location": 0.15,
        "availability": 0.2,
    }


def test_duplicate_provider_names_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "providers.csv"
    source.write_text(
        "name,type,price,currency,expertise,location,availability,portfolio_summary,references,notes\n"
        "Alpha,Agency,100,USD,MCP,Remote,2026-01-01,x,x,x\n"
        "alpha,Agency,100,USD,MCP,Remote,2026-01-01,x,x,x\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate provider names"):
        load_providers(source)


def test_scoring_filters_and_ranks_providers() -> None:
    providers = load_providers(ROOT / "examples" / "providers.csv")
    criteria = load_criteria(ROOT / "config" / "criteria.yaml")

    ranked, excluded = score_providers(providers, criteria)

    assert [item.provider_name for item in ranked] == ["Atlas AI Studio", "Prism Data Collective"]
    assert ranked[0].rank == 1
    assert ranked[0].total_score >= ranked[1].total_score
    assert {item.provider_name for item in excluded} == {"Northstar Automation", "VectorOps Labs"}
    assert any("latest start" in reason for item in excluded for reason in item.exclusions)
    assert any("max budget" in reason for item in excluded for reason in item.exclusions)


def test_missing_fields_receive_penalties() -> None:
    criteria = CriteriaConfig.model_validate(
        {
            "weights": {"price": 1.0, "expertise": 1.0},
            "hard_filters": {"required_expertise": []},
            "penalties": {"missing_price": 0.4},
        }
    )
    providers = [
        ProviderProfile(name="Known", price=100, expertise="MCP", currency="USD"),
        ProviderProfile(name="Unknown", price=None, expertise="MCP", currency="USD"),
    ]

    ranked, excluded = score_providers(providers, criteria)

    assert not excluded
    unknown = next(item for item in ranked if item.provider_name == "Unknown")
    assert "price" in unknown.missing_fields
    price_component = next(
        component for component in unknown.components if component.criterion == "price"
    )
    assert price_component.score == 0.6
