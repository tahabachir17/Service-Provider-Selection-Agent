from __future__ import annotations

from datetime import date
from math import exp
from typing import Any

from provider_selection_agent.models import (
    CriteriaConfig,
    CriterionScore,
    ProviderProfile,
    ProviderScore,
)


def score_providers(
    providers: list[ProviderProfile], criteria: CriteriaConfig
) -> tuple[list[ProviderScore], list[ProviderScore]]:
    eligible = [provider for provider in providers if not _exclusion_reasons(provider, criteria)]
    max_price = max((provider.price or 0.0 for provider in eligible), default=0.0)
    min_price = min(
        (provider.price or max_price for provider in eligible if provider.price),
        default=0.0,
    )

    scored: list[ProviderScore] = []
    excluded: list[ProviderScore] = []
    for provider in providers:
        exclusions = _exclusion_reasons(provider, criteria)
        missing_fields = _missing_fields(provider)
        if exclusions:
            excluded.append(
                ProviderScore(
                    provider_name=provider.name,
                    status="excluded",
                    exclusions=exclusions,
                    missing_fields=missing_fields,
                )
            )
            continue

        components = [
            _score_price(provider, criteria, min_price=min_price, max_price=max_price),
            _score_expertise(provider, criteria),
            _score_location(provider, criteria),
            _score_availability(provider, criteria),
        ]
        total = round(sum(component.weighted_score for component in components), 4)
        scored.append(
            ProviderScore(
                provider_name=provider.name,
                status="eligible",
                total_score=total,
                components=components,
                missing_fields=missing_fields,
            )
        )

    scored.sort(key=lambda item: (-item.total_score, item.provider_name.lower()))
    for index, item in enumerate(scored, start=1):
        item.rank = index
    return scored, excluded


def _exclusion_reasons(provider: ProviderProfile, criteria: CriteriaConfig) -> list[str]:
    filters = criteria.hard_filters
    reasons: list[str] = []

    if filters.currency and provider.currency.lower() != filters.currency.lower():
        reasons.append(f"currency {provider.currency} does not match required {filters.currency}")
    if (
        filters.max_budget is not None
        and provider.price is not None
        and provider.price > filters.max_budget
    ):
        reasons.append(f"price {provider.price:g} exceeds max budget {filters.max_budget:g}")
    if (
        filters.latest_start_date
        and provider.availability
        and provider.availability > filters.latest_start_date
    ):
        reasons.append(
            f"availability {provider.availability.isoformat()} is after latest start "
            f"{filters.latest_start_date.isoformat()}"
        )

    expertise_text = provider.expertise.lower()
    missing_expertise = [
        required
        for required in filters.required_expertise
        if required.lower() not in expertise_text
    ]
    if missing_expertise:
        reasons.append(f"missing required expertise: {', '.join(missing_expertise)}")

    if filters.allowed_locations and not _matches_any(provider.location, filters.allowed_locations):
        reasons.append(f"location {provider.location} is not allowed")

    return reasons


def _missing_fields(provider: ProviderProfile) -> list[str]:
    missing: list[str] = []
    if provider.price is None:
        missing.append("price")
    if provider.expertise == "unknown":
        missing.append("expertise")
    if provider.location == "unknown":
        missing.append("location")
    if provider.availability is None:
        missing.append("availability")
    return missing


def _score_price(
    provider: ProviderProfile, criteria: CriteriaConfig, *, min_price: float, max_price: float
) -> CriterionScore:
    weight = criteria.normalized_weights.get("price", 0.0)
    if provider.price is None:
        score = 1.0 - criteria.penalties.missing_price
        explanation = "Price is unknown, so a missing-data penalty was applied."
    elif max_price <= min_price:
        score = 1.0
        explanation = "Only one eligible known price, so price receives full score."
    else:
        score = 1.0 - ((provider.price - min_price) / (max_price - min_price))
        explanation = "Lower eligible prices receive higher price scores."
    return _component("price", score, weight, explanation, {"price": provider.price})


def _score_expertise(provider: ProviderProfile, criteria: CriteriaConfig) -> CriterionScore:
    weight = criteria.normalized_weights.get("expertise", 0.0)
    required = criteria.hard_filters.required_expertise
    if provider.expertise == "unknown":
        score = 1.0 - criteria.penalties.missing_expertise
        explanation = "Expertise is unknown, so a missing-data penalty was applied."
    elif not required:
        score = 1.0
        explanation = "No required expertise keywords were configured."
    else:
        text = provider.expertise.lower()
        matches = sum(1 for item in required if item.lower() in text)
        score = matches / len(required)
        explanation = f"Matched {matches} of {len(required)} required expertise keywords."
    return _component("expertise", score, weight, explanation, {"expertise": provider.expertise})


def _score_location(provider: ProviderProfile, criteria: CriteriaConfig) -> CriterionScore:
    weight = criteria.normalized_weights.get("location", 0.0)
    allowed = criteria.hard_filters.allowed_locations
    if provider.location == "unknown":
        score = 1.0 - criteria.penalties.missing_location
        explanation = "Location is unknown, so a missing-data penalty was applied."
    elif not allowed:
        score = 1.0
        explanation = "No location preferences were configured."
    elif _matches_any(provider.location, allowed):
        score = 1.0
        explanation = "Location matches an allowed location."
    else:
        score = 0.0
        explanation = "Location does not match the configured allowed locations."
    return _component("location", score, weight, explanation, {"location": provider.location})


def _score_availability(provider: ProviderProfile, criteria: CriteriaConfig) -> CriterionScore:
    weight = criteria.normalized_weights.get("availability", 0.0)
    target = criteria.hard_filters.latest_start_date
    if provider.availability is None:
        score = 1.0 - criteria.penalties.missing_availability
        explanation = "Availability is unknown, so a missing-data penalty was applied."
    elif target is None:
        score = 1.0
        explanation = "No target start date was configured."
    else:
        days_early = (target - provider.availability).days
        score = 1.0 if days_early >= 0 else exp(days_early / 30)
        explanation = "Providers available on or before the latest start date score highest."
    return _component(
        "availability",
        score,
        weight,
        explanation,
        {"availability": _date_iso(provider.availability)},
    )


def _component(
    criterion: str, score: float, weight: float, explanation: str, evidence: dict[str, Any]
) -> CriterionScore:
    clamped = max(0.0, min(1.0, score))
    return CriterionScore(
        criterion=criterion,
        score=round(clamped, 4),
        weight=round(weight, 4),
        weighted_score=round(clamped * weight, 4),
        explanation=explanation,
        evidence=evidence,
    )


def _matches_any(value: str, candidates: list[str]) -> bool:
    text = value.lower()
    return any(candidate.lower() in text or text in candidate.lower() for candidate in candidates)


def _date_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None
