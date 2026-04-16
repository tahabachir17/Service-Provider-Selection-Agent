from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScoreStatus = Literal["eligible", "excluded"]
RunStatus = Literal["DRAFT_PENDING_APPROVAL"]


class ProviderProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    type: str = "unknown"
    price: float | None = None
    currency: str = "unknown"
    expertise: str = "unknown"
    location: str = "unknown"
    availability: date | None = None
    portfolio_summary: str = "unknown"
    references: str = "unknown"
    notes: str = "unknown"

    @field_validator("name")
    @classmethod
    def name_required(cls, value: str) -> str:
        if not value:
            raise ValueError("provider name is required")
        return value

    @field_validator("price", mode="before")
    @classmethod
    def parse_price(cls, value: Any) -> float | None:
        if value in (None, "", "unknown"):
            return None
        return float(value)

    @field_validator("availability", mode="before")
    @classmethod
    def parse_availability(cls, value: Any) -> date | None:
        if value in (None, "", "unknown"):
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))


class HardFilters(BaseModel):
    max_budget: float | None = None
    currency: str | None = None
    required_expertise: list[str] = Field(default_factory=list)
    allowed_locations: list[str] = Field(default_factory=list)
    latest_start_date: date | None = None


class Penalties(BaseModel):
    missing_price: float = 0.25
    missing_expertise: float = 0.20
    missing_location: float = 0.10
    missing_availability: float = 0.20


class ReportConfig(BaseModel):
    top_n: int = 3
    require_human_approval: bool = True


class CriteriaConfig(BaseModel):
    weights: dict[str, float]
    hard_filters: HardFilters = Field(default_factory=HardFilters)
    penalties: Penalties = Field(default_factory=Penalties)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("weights")
    @classmethod
    def supported_weights(cls, value: dict[str, float]) -> dict[str, float]:
        supported = {"price", "expertise", "location", "availability"}
        unknown = set(value) - supported
        if unknown:
            raise ValueError(f"unsupported scoring weights: {sorted(unknown)}")
        if not value or sum(value.values()) <= 0:
            raise ValueError("at least one positive weight is required")
        return value

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(max(weight, 0.0) for weight in self.weights.values())
        return {name: max(weight, 0.0) / total for name, weight in self.weights.items()}


class CriterionScore(BaseModel):
    criterion: str
    score: float
    weight: float
    weighted_score: float
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ProviderScore(BaseModel):
    provider_name: str
    status: ScoreStatus
    total_score: float = 0.0
    rank: int | None = None
    components: list[CriterionScore] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class ProviderComparison(BaseModel):
    provider_name: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ComparisonSynthesis(BaseModel):
    executive_summary: str
    recommended_provider: str | None = None
    rationale: str
    comparisons: list[ProviderComparison] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None


class RecommendationReport(BaseModel):
    status: RunStatus = "DRAFT_PENDING_APPROVAL"
    synthesis: ComparisonSynthesis
    ranked_scores: list[ProviderScore]
    excluded_scores: list[ProviderScore] = Field(default_factory=list)


class WorkflowTraceStep(BaseModel):
    step_name: str
    title: str
    details: str
    snapshot: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    providers_path: str
    criteria_path: str
    brief_path: str
    output_dir: str
    providers: list[ProviderProfile] = Field(default_factory=list)
    criteria: CriteriaConfig | None = None
    project_brief: str = ""
    ranked_scores: list[ProviderScore] = Field(default_factory=list)
    excluded_scores: list[ProviderScore] = Field(default_factory=list)
    synthesis: ComparisonSynthesis | None = None
    report: RecommendationReport | None = None
    audit: dict[str, Any] = Field(default_factory=dict)
    trace_steps: list[WorkflowTraceStep] = Field(default_factory=list)
