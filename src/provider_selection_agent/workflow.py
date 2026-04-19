from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from provider_selection_agent.config import Settings
from provider_selection_agent.llm import SynthesisClient, synthesize_comparison
from provider_selection_agent.loaders import load_brief, load_criteria, load_providers
from provider_selection_agent.mcp import MCPConnectorConfig, MCPProviderEnrichmentClient
from provider_selection_agent.models import (
    RecommendationReport,
    WorkflowState,
    WorkflowTraceStep,
)
from provider_selection_agent.reporting import write_outputs
from provider_selection_agent.scoring import score_providers

StateDict = dict[str, Any]
TraceObserver = Callable[[WorkflowTraceStep, WorkflowState], None]


def run_workflow(
    *,
    providers_path: str,
    criteria_path: str,
    brief_path: str,
    output_dir: str,
    settings: Settings,
    synthesis_client: SynthesisClient | None = None,
) -> WorkflowState:
    state = WorkflowState(
        providers_path=providers_path,
        criteria_path=criteria_path,
        brief_path=brief_path,
        output_dir=output_dir,
    )
    graph = _build_graph(settings=settings, synthesis_client=synthesis_client)
    result = graph.invoke(state.model_dump(mode="json"))
    return WorkflowState.model_validate(result)


def run_workflow_traced(
    *,
    providers_path: str,
    criteria_path: str,
    brief_path: str,
    output_dir: str,
    settings: Settings,
    synthesis_client: SynthesisClient | None = None,
    observer: TraceObserver | None = None,
) -> WorkflowState:
    state = WorkflowState(
        providers_path=providers_path,
        criteria_path=criteria_path,
        brief_path=brief_path,
        output_dir=output_dir,
    )
    nodes: list[tuple[str, Callable[[StateDict], StateDict]]] = [
        ("load_inputs", _load_inputs),
        ("validate_profiles", _validate_profiles),
        ("normalize_profiles", _normalize_profiles),
        ("enrich_provider_data", lambda raw_state: _enrich_provider_data(raw_state, settings)),
        ("score_candidates", _score_candidates),
        (
            "llm_compare_top_candidates",
            lambda raw_state: _llm_compare_top_candidates(
                raw_state,
                settings=settings,
                synthesis_client=synthesis_client,
            ),
        ),
        ("generate_draft_report", _generate_draft_report),
        ("human_review_gate", _human_review_gate),
        ("write_outputs", _write_outputs),
    ]
    raw_state = state.model_dump(mode="json")
    for name, node in nodes:
        raw_state = node(raw_state)
        state = WorkflowState.model_validate(raw_state)
        trace_step = _trace_step_for(name, state)
        state.trace_steps.append(trace_step)
        raw_state = state.model_dump(mode="json")
        if observer is not None:
            observer(trace_step, state)
    return state


def _build_graph(settings: Settings, synthesis_client: SynthesisClient | None) -> Any:
    nodes: list[tuple[str, Callable[[StateDict], StateDict]]] = [
        ("load_inputs", _load_inputs),
        ("validate_profiles", _validate_profiles),
        ("normalize_profiles", _normalize_profiles),
        ("enrich_provider_data", lambda state: _enrich_provider_data(state, settings)),
        ("score_candidates", _score_candidates),
        (
            "llm_compare_top_candidates",
            lambda state: _llm_compare_top_candidates(
                state, settings=settings, synthesis_client=synthesis_client
            ),
        ),
        ("generate_draft_report", _generate_draft_report),
        ("human_review_gate", _human_review_gate),
        ("write_outputs", _write_outputs),
    ]
    try:
        from langgraph.graph import END, StateGraph

        graph_factory = cast(Any, StateGraph)
        end_node = cast(Any, END)
        graph = graph_factory(dict)
        for name, node in nodes:
            graph.add_node(name, node)
        graph.set_entry_point(nodes[0][0])
        for (current, _), (following, _) in zip(nodes, nodes[1:], strict=False):
            graph.add_edge(current, following)
        graph.add_edge(nodes[-1][0], end_node)
        return graph.compile()
    except Exception:
        return SequentialGraph(nodes)


class SequentialGraph:
    def __init__(self, nodes: list[tuple[str, Callable[[StateDict], StateDict]]]) -> None:
        self.nodes = nodes

    def invoke(self, state: StateDict) -> StateDict:
        for _, node in self.nodes:
            state = node(state)
        return state


def _load_inputs(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    state.providers = load_providers(state.providers_path)
    state.criteria = load_criteria(state.criteria_path)
    state.project_brief = load_brief(state.brief_path)
    state.audit["loaded_provider_count"] = len(state.providers)
    return state.model_dump(mode="json")


def _validate_profiles(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    if not state.providers:
        raise ValueError("at least one provider is required")
    state.audit["validated_provider_count"] = len(state.providers)
    return state.model_dump(mode="json")


def _normalize_profiles(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    state.audit["normalization"] = "pydantic models normalized missing values to unknown/null"
    return state.model_dump(mode="json")


def _score_candidates(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    if state.criteria is None:
        raise ValueError("criteria must be loaded before scoring")
    state.ranked_scores, state.excluded_scores = score_providers(state.providers, state.criteria)
    state.audit["eligible_count"] = len(state.ranked_scores)
    state.audit["excluded_count"] = len(state.excluded_scores)
    return state.model_dump(mode="json")


def _enrich_provider_data(raw_state: StateDict, settings: Settings) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    client = MCPProviderEnrichmentClient(MCPConnectorConfig.from_settings(settings))
    try:
        state.providers, enrichment_audit = client.enrich_providers(state.providers)
    except Exception as exc:
        enrichment_audit = {
            "attempted": True,
            "failed": True,
            "reason": str(exc),
            "applied_updates": [],
        }
    state.audit["mcp_enrichment"] = enrichment_audit
    return state.model_dump(mode="json")


def _llm_compare_top_candidates(
    raw_state: StateDict, *, settings: Settings, synthesis_client: SynthesisClient | None
) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    if state.criteria is None:
        raise ValueError("criteria must be loaded before LLM synthesis")
    state.synthesis = synthesize_comparison(
        project_brief=state.project_brief,
        providers=state.providers,
        ranked_scores=state.ranked_scores,
        settings=settings,
        top_n=state.criteria.report.top_n,
        client=synthesis_client,
    )
    return state.model_dump(mode="json")


def _generate_draft_report(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    if state.synthesis is None:
        raise ValueError("synthesis is required before report generation")
    state.report = RecommendationReport(
        synthesis=state.synthesis,
        ranked_scores=state.ranked_scores,
        excluded_scores=state.excluded_scores,
    )
    return state.model_dump(mode="json")


def _human_review_gate(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    if state.report is None:
        raise ValueError("report is required before human review gate")
    state.report.status = "DRAFT_PENDING_APPROVAL"
    state.audit["human_review_required"] = True
    return state.model_dump(mode="json")


def _write_outputs(raw_state: StateDict) -> StateDict:
    state = WorkflowState.model_validate(raw_state)
    write_outputs(state)
    return state.model_dump(mode="json")


def _trace_step_for(step_name: str, state: WorkflowState) -> WorkflowTraceStep:
    if step_name == "load_inputs":
        return WorkflowTraceStep(
            step_name=step_name,
            title="Load Input Files",
            details=(
                f"Loaded {len(state.providers)} provider profiles, criteria configuration, "
                "and the project brief."
            ),
            snapshot={
                "provider_count": len(state.providers),
                "criteria_loaded": state.criteria is not None,
                "brief_loaded": bool(state.project_brief),
            },
        )
    if step_name == "validate_profiles":
        return WorkflowTraceStep(
            step_name=step_name,
            title="Validate Profiles",
            details="Validated typed provider records and rejected structural issues.",
            snapshot={
                "validated_provider_count": len(state.providers),
            },
        )
    if step_name == "normalize_profiles":
        return WorkflowTraceStep(
            step_name=step_name,
            title="Normalize Profiles",
            details=(
                "Normalized missing values into explicit unknown/null states "
                "so scoring stays auditable."
            ),
            snapshot={
                "normalization": state.audit.get("normalization"),
            },
        )
    if step_name == "score_candidates":
        return WorkflowTraceStep(
            step_name=step_name,
            title="Score Candidates",
            details=(
                f"Applied hard filters, ranked {len(state.ranked_scores)} eligible providers, "
                f"and excluded {len(state.excluded_scores)}."
            ),
            snapshot={
                "eligible": [score.provider_name for score in state.ranked_scores],
                "excluded": [
                    {
                        "provider_name": score.provider_name,
                        "exclusions": score.exclusions,
                    }
                    for score in state.excluded_scores
                ],
            },
        )
    if step_name == "enrich_provider_data":
        enrichment = state.audit.get("mcp_enrichment", {})
        return WorkflowTraceStep(
            step_name=step_name,
            title="Enrich Provider Data",
            details=(
                "Fetched provider field updates from the MCP connector."
                if enrichment.get("attempted") and not enrichment.get("failed")
                else (
                    "Skipped MCP enrichment because no connector was configured."
                    if not enrichment.get("attempted")
                    else "MCP enrichment failed and the workflow continued with original data."
                )
            ),
            snapshot={
                "attempted": enrichment.get("attempted", False),
                "failed": enrichment.get("failed", False),
                "updated_provider_count": enrichment.get("updated_provider_count", 0),
                "wanted_fields": enrichment.get("wanted_fields", []),
            },
        )
    if step_name == "llm_compare_top_candidates":
        fallback_reason = state.synthesis.fallback_reason if state.synthesis else None
        return WorkflowTraceStep(
            step_name=step_name,
            title="Synthesize Tradeoffs",
            details=(
                "Used the LLM for qualitative comparison."
                if state.synthesis and not state.synthesis.fallback_used
                else (
                    "Skipped to deterministic fallback synthesis because the LLM "
                    "was unavailable or rejected."
                )
            ),
            snapshot={
                "recommended_provider": (
                    state.synthesis.recommended_provider if state.synthesis else None
                ),
                "fallback_used": state.synthesis.fallback_used if state.synthesis else True,
                "fallback_reason": fallback_reason,
            },
        )
    if step_name == "generate_draft_report":
        return WorkflowTraceStep(
            step_name=step_name,
            title="Generate Draft Report",
            details=(
                "Built the stakeholder-facing draft recommendation and "
                "structured audit payload."
            ),
            snapshot={
                "status": state.report.status if state.report else None,
                "recommended_provider": (
                    state.report.synthesis.recommended_provider if state.report else None
                ),
            },
        )
    if step_name == "human_review_gate":
        return WorkflowTraceStep(
            step_name=step_name,
            title="Require Human Approval",
            details="Marked the recommendation as draft-only pending stakeholder approval.",
            snapshot={
                "status": state.report.status if state.report else None,
                "human_review_required": state.audit.get("human_review_required", True),
            },
        )
    return WorkflowTraceStep(
        step_name=step_name,
        title="Write Outputs",
        details="Wrote recommendation.md, scores.json, audit.json, and approval_required.txt.",
        snapshot={
            "output_dir": state.output_dir,
        },
    )
