from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from provider_selection_agent.models import WorkflowState

REPORT_TEMPLATE = Template(
    """# Service Provider Recommendation

Status: **{{ report.status }}**

{% if report.synthesis.fallback_used -%}
Fallback mode: **yes**  
Reason: {{ report.synthesis.fallback_reason }}
{% else -%}
Fallback mode: **no**
{% endif %}

## Executive Summary

{{ report.synthesis.executive_summary }}

## Recommended Provider

{{ report.synthesis.recommended_provider or "No eligible provider" }}

## Rationale

{{ report.synthesis.rationale }}

## Ranked Candidates

{% for score in report.ranked_scores -%}
### {{ score.rank }}. {{ score.provider_name }} - {{ "%.2f"|format(score.total_score) }}
{% if score.missing_fields -%}
Missing data: {{ score.missing_fields|join(", ") }}
{% endif -%}
{% for component in score.components -%}
- {{ component.criterion }}:
  score {{ "%.2f"|format(component.score) }},
  weight {{ "%.2f"|format(component.weight) }},
  weighted {{ "%.2f"|format(component.weighted_score) }}.
  {{ component.explanation }}
{% endfor %}
{% endfor %}

## Qualitative Comparison

{% for item in report.synthesis.comparisons -%}
### {{ item.provider_name }}
Strengths: {{ item.strengths|join("; ") if item.strengths else "None recorded" }}

Weaknesses: {{ item.weaknesses|join("; ") if item.weaknesses else "None recorded" }}

Risks: {{ item.risks|join("; ") if item.risks else "None recorded" }}

Evidence: {{ item.evidence_refs|join(", ") if item.evidence_refs else "None" }}
{% endfor %}

## Excluded Providers

{% if report.excluded_scores -%}
{% for score in report.excluded_scores -%}
- {{ score.provider_name }}: {{ score.exclusions|join("; ") }}
{% endfor %}
{% else -%}
No providers were excluded.
{% endif %}

## Approval

This report is a draft and requires stakeholder approval before final selection.
"""
)


def write_outputs(state: WorkflowState) -> None:
    if state.report is None or state.criteria is None:
        raise ValueError("workflow state is missing report or criteria")

    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_md = REPORT_TEMPLATE.render(report=state.report)
    (output_dir / "recommendation.md").write_text(report_md, encoding="utf-8")
    (output_dir / "scores.json").write_text(
        _json_dump(
            {
                "ranked_scores": [score.model_dump(mode="json") for score in state.ranked_scores],
                "excluded_scores": [
                    score.model_dump(mode="json") for score in state.excluded_scores
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "audit.json").write_text(_json_dump(_audit_payload(state)), encoding="utf-8")
    (output_dir / "approval_required.txt").write_text(
        "DRAFT_PENDING_APPROVAL: Human approval is required before final provider selection.\n",
        encoding="utf-8",
    )


def _audit_payload(state: WorkflowState) -> dict[str, Any]:
    criteria = state.criteria
    report = state.report
    if criteria is None or report is None:
        raise ValueError("workflow state is missing report or criteria")
    return {
        "inputs": {
            "providers_path": state.providers_path,
            "criteria_path": state.criteria_path,
            "brief_path": state.brief_path,
        },
        "weights": criteria.normalized_weights,
        "hard_filters": criteria.hard_filters.model_dump(mode="json"),
        "mcp_enrichment": state.audit.get("mcp_enrichment", {}),
        "ranked_scores": [score.model_dump(mode="json") for score in state.ranked_scores],
        "excluded_scores": [score.model_dump(mode="json") for score in state.excluded_scores],
        "llm_evidence_refs": report.synthesis.evidence_refs,
        "status": report.status,
        "fallback_used": report.synthesis.fallback_used,
        "fallback_reason": report.synthesis.fallback_reason,
    }


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
