from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

from pydantic import SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential

from provider_selection_agent.config import Settings
from provider_selection_agent.models import (
    ComparisonSynthesis,
    ProviderComparison,
    ProviderProfile,
    ProviderScore,
)

SynthesisClient = Callable[
    [str, list[ProviderProfile], list[ProviderScore], int],
    ComparisonSynthesis,
]


def synthesize_comparison(
    *,
    project_brief: str,
    providers: list[ProviderProfile],
    ranked_scores: list[ProviderScore],
    settings: Settings,
    top_n: int,
    client: SynthesisClient | None = None,
) -> ComparisonSynthesis:
    if client is not None:
        return _validated_or_fallback(
            client(project_brief, providers, ranked_scores, top_n),
            providers,
        )

    if not settings.llm_api_key:
        return score_only_synthesis(
            ranked_scores,
            reason=f"{settings.api_key_env_hint} is not configured",
        )

    try:
        synthesis = _call_llm_structured(
            project_brief,
            providers,
            ranked_scores,
            settings,
            top_n,
        )
        return _validated_or_fallback(synthesis, providers)
    except Exception as exc:  # pragma: no cover - live LLM failure path
        return score_only_synthesis(ranked_scores, reason=f"LLM synthesis failed: {exc}")


def score_only_synthesis(ranked_scores: list[ProviderScore], *, reason: str) -> ComparisonSynthesis:
    winner = ranked_scores[0].provider_name if ranked_scores else None
    comparisons = [
        ProviderComparison(
            provider_name=item.provider_name,
            strengths=[f"Ranked #{item.rank} with deterministic score {item.total_score:.2f}."],
            weaknesses=["No LLM qualitative synthesis was used; review component scores manually."],
            risks=item.missing_fields or [],
            evidence_refs=[f"{item.provider_name}:score"],
        )
        for item in ranked_scores[:3]
    ]
    if winner:
        summary = f"{winner} is the top deterministic candidate by weighted score."
        rationale = (
            "The fallback recommendation is based only on configured weights and hard filters."
        )
    else:
        summary = "No eligible providers were found."
        rationale = "All providers were excluded by configured hard filters."
    return ComparisonSynthesis(
        executive_summary=summary,
        recommended_provider=winner,
        rationale=rationale,
        comparisons=comparisons,
        evidence_refs=[ref for comparison in comparisons for ref in comparison.evidence_refs],
        fallback_used=True,
        fallback_reason=reason,
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def _call_llm_structured(
    project_brief: str,
    providers: list[ProviderProfile],
    ranked_scores: list[ProviderScore],
    settings: Settings,
    top_n: int,
) -> ComparisonSynthesis:
    from langchain_openai import ChatOpenAI

    if settings.llm_provider == "gemini":
        api_key = settings.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    elif settings.llm_provider == "groq":
        api_key = settings.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    else:
        api_key = settings.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(f"{settings.api_key_env_hint} is not configured")

    chat_model: Any
    if settings.llm_base_url:
        chat_model = ChatOpenAI(
            model=settings.llm_model,
            api_key=SecretStr(api_key),
            temperature=0,
            base_url=settings.llm_base_url,
        )
    else:
        chat_model = ChatOpenAI(
            model=settings.llm_model,
            api_key=SecretStr(api_key),
            temperature=0,
        )

    model = cast(Any, chat_model).with_structured_output(ComparisonSynthesis)

    provider_records = [provider.model_dump(mode="json") for provider in providers]
    score_records = [score.model_dump(mode="json") for score in ranked_scores[:top_n]]
    prompt = (
        "You are drafting a service provider selection recommendation. "
        "Use only the provided provider records and score records. "
        "Do not invent evidence. Numeric scores are final and cannot be changed. "
        "Every evidence reference must use '<provider name>:<field>' "
        "or '<provider name>:score'. "
        f"The configured LLM provider is '{settings.llm_provider}'.\n\n"
        f"Project brief:\n{project_brief}\n\n"
        f"Provider records:\n{provider_records}\n\n"
        f"Ranked scores:\n{score_records}\n"
    )
    result = model.invoke(prompt)
    if not isinstance(result, ComparisonSynthesis):
        return ComparisonSynthesis.model_validate(result)
    return result


def _validated_or_fallback(
    synthesis: ComparisonSynthesis, providers: list[ProviderProfile]
) -> ComparisonSynthesis:
    valid_refs = _valid_evidence_refs(providers)
    invalid = [ref for ref in synthesis.evidence_refs if ref not in valid_refs]
    for comparison in synthesis.comparisons:
        invalid.extend(ref for ref in comparison.evidence_refs if ref not in valid_refs)
    if invalid:
        return ComparisonSynthesis(
            executive_summary="LLM synthesis was discarded because it cited unavailable evidence.",
            recommended_provider=None,
            rationale=(
                "The recommendation requires human review because evidence validation failed."
            ),
            comparisons=[],
            evidence_refs=[],
            fallback_used=True,
            fallback_reason=f"Invalid evidence references: {sorted(set(invalid))}",
        )
    return synthesis


def _valid_evidence_refs(providers: list[ProviderProfile]) -> set[str]:
    fields = {
        "type",
        "price",
        "currency",
        "expertise",
        "location",
        "availability",
        "portfolio_summary",
        "references",
        "notes",
        "score",
    }
    return {f"{provider.name}:{field}" for provider in providers for field in fields}
