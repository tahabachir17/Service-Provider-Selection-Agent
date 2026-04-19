from __future__ import annotations

import json
from pathlib import Path

from provider_selection_agent.config import (
    GEMINI_OPENAI_BASE_URL,
    GROQ_OPENAI_BASE_URL,
    Settings,
    load_settings,
)
from provider_selection_agent.llm import synthesize_comparison
from provider_selection_agent.models import (
    ComparisonSynthesis,
    ProviderComparison,
    ProviderProfile,
    ProviderScore,
)
from provider_selection_agent.workflow import run_workflow, run_workflow_traced

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_api_key=None,
        llm_model="test-model",
        llm_base_url=None,
        log_level="INFO",
        vector_db_path=".local/vector_store",
        enable_web_search=False,
        mcp_server_url=None,
        mcp_enrich_fields=("price", "expertise", "location", "availability"),
        mcp_timeout_seconds=120,
    )


def test_synthesis_falls_back_without_openai_key() -> None:
    result = synthesize_comparison(
        project_brief="brief",
        providers=[ProviderProfile(name="Alpha")],
        ranked_scores=[
            ProviderScore(provider_name="Alpha", status="eligible", total_score=1, rank=1)
        ],
        settings=_settings(),
        top_n=1,
    )

    assert result.fallback_used is True
    assert result.recommended_provider == "Alpha"
    assert "OPENAI_API_KEY" in (result.fallback_reason or "")


def test_invalid_llm_evidence_is_rejected() -> None:
    def fake_client(_brief, _providers, _ranked, _top_n):
        return ComparisonSynthesis(
            executive_summary="summary",
            recommended_provider="Alpha",
            rationale="rationale",
            comparisons=[
                ProviderComparison(
                    provider_name="Alpha",
                    strengths=["strong"],
                    evidence_refs=["Alpha:made_up_field"],
                )
            ],
            evidence_refs=["Alpha:made_up_field"],
        )

    result = synthesize_comparison(
        project_brief="brief",
        providers=[ProviderProfile(name="Alpha")],
        ranked_scores=[
            ProviderScore(provider_name="Alpha", status="eligible", total_score=1, rank=1)
        ],
        settings=_settings(),
        top_n=1,
        client=fake_client,
    )

    assert result.fallback_used is True
    assert "Invalid evidence references" in (result.fallback_reason or "")


def test_workflow_writes_expected_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"

    state = run_workflow(
        providers_path=str(ROOT / "examples" / "providers.csv"),
        criteria_path=str(ROOT / "config" / "criteria.yaml"),
        brief_path=str(ROOT / "examples" / "project_brief.md"),
        output_dir=str(out_dir),
        settings=_settings(),
    )

    assert state.report is not None
    assert state.report.status == "DRAFT_PENDING_APPROVAL"
    assert (out_dir / "recommendation.md").exists()
    assert (out_dir / "scores.json").exists()
    assert (out_dir / "audit.json").exists()
    assert (out_dir / "approval_required.txt").exists()

    audit = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "DRAFT_PENDING_APPROVAL"
    assert audit["human_review_required"] if "human_review_required" in audit else True
    assert audit["fallback_used"] is True


def test_traced_workflow_emits_all_steps(tmp_path: Path) -> None:
    observed_steps: list[str] = []

    def observer(trace_step, _state) -> None:
        observed_steps.append(trace_step.step_name)

    state = run_workflow_traced(
        providers_path=str(ROOT / "examples" / "providers.csv"),
        criteria_path=str(ROOT / "config" / "criteria.yaml"),
        brief_path=str(ROOT / "examples" / "project_brief.md"),
        output_dir=str(tmp_path / "trace_run"),
        settings=_settings(),
        observer=observer,
    )

    assert observed_steps == [
        "load_inputs",
        "validate_profiles",
        "normalize_profiles",
        "enrich_provider_data",
        "score_candidates",
        "llm_compare_top_candidates",
        "generate_draft_report",
        "human_review_gate",
        "write_outputs",
    ]
    assert len(state.trace_steps) == 9


def test_load_settings_supports_gemini_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    settings = load_settings(load_env_file=False)

    assert settings.llm_provider == "gemini"
    assert settings.llm_api_key == "test-gemini-key"
    assert settings.llm_model == "gemini-2.5-flash"
    assert settings.llm_base_url == GEMINI_OPENAI_BASE_URL
    assert settings.mcp_enrich_fields == ("price", "expertise", "location", "availability")
    assert settings.mcp_timeout_seconds == 300


def test_load_settings_supports_groq_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    settings = load_settings(load_env_file=False)

    assert settings.llm_provider == "groq"
    assert settings.llm_api_key == "test-groq-key"
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.llm_base_url == GROQ_OPENAI_BASE_URL
    assert settings.mcp_timeout_seconds == 300
