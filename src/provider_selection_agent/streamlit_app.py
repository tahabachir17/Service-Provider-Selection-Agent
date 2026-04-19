from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from provider_selection_agent.config import load_settings  # noqa: E402
from provider_selection_agent.models import WorkflowState, WorkflowTraceStep  # noqa: E402
from provider_selection_agent.sourcing import (  # noqa: E402
    discover_providers_via_mcp,
    write_discovery_output,
)
from provider_selection_agent.workflow import run_workflow_traced  # noqa: E402

APP_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROVIDERS = APP_ROOT / "examples" / "providers.csv"
EXAMPLE_CRITERIA = APP_ROOT / "config" / "criteria.yaml"
EXAMPLE_BRIEF = APP_ROOT / "examples" / "project_brief.md"
REPORTS_ROOT = APP_ROOT / "reports"


def main() -> None:
    st.set_page_config(
        page_title="Service Provider Selection Agent",
        page_icon=":material/account_tree:",
        layout="wide",
    )
    _inject_styles()

    st.title("Service Provider Selection Agent")
    st.caption(
        "Deterministic provider ranking with a visible agent workflow, audit trail, "
        "and draft recommendation."
    )

    with st.sidebar:
        st.subheader("Run Setup")
        use_examples = st.toggle("Use bundled example files", value=True)
        discover_via_mcp = st.toggle("Discover providers via MCP", value=False)
        uploaded_providers_disabled = use_examples or discover_via_mcp
        uploaded_providers = st.file_uploader(
            "Providers CSV or JSON",
            type=["csv", "json"],
            disabled=uploaded_providers_disabled,
        )
        uploaded_criteria = st.file_uploader(
            "Criteria YAML",
            type=["yaml", "yml"],
            disabled=use_examples,
        )
        uploaded_brief = st.file_uploader(
            "Project brief",
            type=["md", "txt"],
            disabled=use_examples,
        )
        if discover_via_mcp:
            st.divider()
            st.caption("MCP Discovery")
            discovery_context = st.text_area(
                "Project context",
                value=(
                    "We are looking for a service provider that fits the project needs "
                    "described in the brief and can support delivery successfully."
                ),
                height=140,
            )
            discovery_target_fields = st.text_input(
                "Target fields",
                value="Full-Stack Web Development, Data Architecture",
                help="Enter a comma-separated list of service capabilities to source for.",
            )
            discovery_preferred_location = st.text_input(
                "Preferred location",
                value="EMEA",
            )
            discovery_remote_ok = st.checkbox(
                "Allow remote providers outside the preferred location",
                value=True,
            )
            discovery_max_results = st.slider(
                "Providers to discover",
                min_value=1,
                max_value=10,
                value=5,
            )
        output_name = st.text_input(
            "Output folder name",
            value=f"streamlit_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        run_clicked = st.button(
            "Run Agent",
            type="primary",
            use_container_width=True,
        )

        settings = load_settings()
        st.divider()
        st.caption("Runtime")
        st.write(f"Provider: `{settings.llm_provider}`")
        st.write(f"Model: `{settings.llm_model}`")
        st.write(
            "LLM mode: "
            + ("enabled" if settings.llm_api_key else "deterministic fallback only")
        )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Inputs")
        if discover_via_mcp:
            _preview_discovery_inputs(
                use_examples=use_examples,
                uploaded_criteria=uploaded_criteria,
                uploaded_brief=uploaded_brief,
                project_context=discovery_context,
                target_fields=discovery_target_fields,
                preferred_location=discovery_preferred_location,
                remote_ok=discovery_remote_ok,
                max_results=discovery_max_results,
            )
        elif use_examples:
            st.success("Using bundled example files.")
            _preview_example_inputs()
        else:
            _preview_uploaded_inputs(uploaded_providers, uploaded_criteria, uploaded_brief)

    with right:
        st.subheader("Agent Workflow")
        workflow_placeholder = st.container()
        result_placeholder = st.container()

    if not run_clicked:
        with workflow_placeholder:
            st.info(
                "Choose inputs and click Run Agent to watch the workflow execute."
                if not discover_via_mcp
                else "Configure the MCP discovery request and click Run Agent to source and rank providers."
            )
        return

    if discover_via_mcp:
        if not discovery_context.strip():
            with workflow_placeholder:
                st.error("Provide project context before running MCP discovery.")
            return
        if not _parse_target_fields(discovery_target_fields):
            with workflow_placeholder:
                st.error("Provide at least one target field for MCP discovery.")
            return
    elif not use_examples and not all([uploaded_providers, uploaded_criteria, uploaded_brief]):
        with workflow_placeholder:
            st.error("Upload providers, criteria, and brief files before running the agent.")
        return

    if not use_examples and not all([uploaded_criteria, uploaded_brief]):
        with workflow_placeholder:
            st.error("Upload criteria and brief files before running the agent.")
        return

    output_dir = REPORTS_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        criteria_path, brief_path = _materialize_non_provider_inputs(
            temp_root,
            use_examples=use_examples,
            uploaded_criteria=uploaded_criteria,
            uploaded_brief=uploaded_brief,
        )
        if discover_via_mcp:
            providers_path = _discover_providers_for_run(
                workflow_placeholder=workflow_placeholder,
                output_dir=output_dir,
                project_context=discovery_context,
                target_fields=_parse_target_fields(discovery_target_fields),
                preferred_location=discovery_preferred_location,
                remote_ok=discovery_remote_ok,
                max_results=discovery_max_results,
            )
            if providers_path is None:
                return
        else:
            providers_path = _materialize_uploaded_providers(
                temp_root,
                use_examples=use_examples,
                uploaded_providers=uploaded_providers,
            )
        _run_with_trace(
            workflow_placeholder=workflow_placeholder,
            result_placeholder=result_placeholder,
            providers_path=providers_path,
            criteria_path=criteria_path,
            brief_path=brief_path,
            output_dir=output_dir,
        )


def _run_with_trace(
    *,
    workflow_placeholder: st.delta_generator.DeltaGenerator,
    result_placeholder: st.delta_generator.DeltaGenerator,
    providers_path: Path,
    criteria_path: Path,
    brief_path: Path,
    output_dir: Path,
) -> None:
    trace_steps: list[dict[str, object]] = []
    progress_bar = workflow_placeholder.progress(0, text="Starting workflow...")
    status_box = workflow_placeholder.empty()
    trace_box = workflow_placeholder.container()
    node_count = 9

    def observer(trace_step: WorkflowTraceStep, _state: WorkflowState) -> None:
        trace_steps.append(trace_step.model_dump(mode="json"))
        current = len(trace_steps)
        progress_bar.progress(
            current / node_count,
            text=f"Step {current}/{node_count}: {trace_step.title}",
        )
        with trace_box:
            st.markdown("#### Execution Trace")
            for index, step in enumerate(trace_steps, start=1):
                with st.expander(f"{index}. {step['title']}", expanded=index == current):
                    st.write(step["details"])
                    st.json(step["snapshot"])

    try:
        status_box.info("Running agent...")
        state = run_workflow_traced(
            providers_path=str(providers_path),
            criteria_path=str(criteria_path),
            brief_path=str(brief_path),
            output_dir=str(output_dir),
            settings=load_settings(),
            observer=observer,
        )
        progress_bar.progress(1.0, text="Workflow complete")
        status_box.success("Agent finished successfully.")
        _render_results(result_placeholder, state, output_dir)
    except Exception as exc:
        progress_bar.empty()
        status_box.error(f"Agent run failed: {exc}")


def _render_results(
    result_placeholder: st.delta_generator.DeltaGenerator,
    state: WorkflowState,
    output_dir: Path,
) -> None:
    report = state.report
    if report is None:
        result_placeholder.error("No report was generated.")
        return

    with result_placeholder:
        st.subheader("Recommendation")
        winner = report.synthesis.recommended_provider or "No eligible provider"
        metric_cols = st.columns(3)
        metric_cols[0].metric("Recommended Provider", winner)
        metric_cols[1].metric("Eligible Providers", len(state.ranked_scores))
        metric_cols[2].metric("Excluded Providers", len(state.excluded_scores))

        st.markdown(report.synthesis.executive_summary)
        if report.synthesis.fallback_used:
            st.warning(
                "The qualitative synthesis used deterministic fallback mode. "
                f"Reason: {report.synthesis.fallback_reason}"
            )
        else:
            st.success("LLM synthesis completed successfully.")

        st.markdown("#### Ranked Providers")
        score_frame = pd.DataFrame(
            [
                {
                    "Rank": score.rank,
                    "Provider": score.provider_name,
                    "Total Score": score.total_score,
                    "Missing Fields": ", ".join(score.missing_fields) or "-",
                }
                for score in state.ranked_scores
            ]
        )
        st.dataframe(score_frame, use_container_width=True, hide_index=True)

        st.markdown("#### Exclusions")
        if state.excluded_scores:
            excluded_frame = pd.DataFrame(
                [
                    {
                        "Provider": score.provider_name,
                        "Reasons": "; ".join(score.exclusions),
                    }
                    for score in state.excluded_scores
                ]
            )
            st.dataframe(excluded_frame, use_container_width=True, hide_index=True)
        else:
            st.caption("No providers were excluded.")

        st.markdown("#### Generated Files")
        filenames = [
            "recommendation.md",
            "scores.json",
            "audit.json",
            "approval_required.txt",
        ]
        if (output_dir / "discovered_providers.json").exists():
            filenames.append("discovered_providers.json")
        artifact_tabs = st.tabs(filenames)
        for tab, filename in zip(artifact_tabs, filenames, strict=True):
            path = output_dir / filename
            content = path.read_text(encoding="utf-8")
            with tab:
                if filename.endswith(".json"):
                    st.json(json.loads(content))
                else:
                    st.code(content, language="markdown")
def _materialize_non_provider_inputs(
    temp_root: Path,
    *,
    use_examples: bool,
    uploaded_criteria: UploadedFile | None,
    uploaded_brief: UploadedFile | None,
) -> tuple[Path, Path]:
    if use_examples:
        return EXAMPLE_CRITERIA, EXAMPLE_BRIEF

    criteria_path = temp_root / uploaded_criteria.name
    brief_path = temp_root / uploaded_brief.name
    criteria_path.write_bytes(uploaded_criteria.getvalue())
    brief_path.write_bytes(uploaded_brief.getvalue())
    return criteria_path, brief_path


def _materialize_uploaded_providers(
    temp_root: Path,
    *,
    use_examples: bool,
    uploaded_providers: UploadedFile | None,
) -> Path:
    if use_examples:
        return EXAMPLE_PROVIDERS

    providers_path = temp_root / uploaded_providers.name
    providers_path.write_bytes(uploaded_providers.getvalue())
    return providers_path


def _discover_providers_for_run(
    *,
    workflow_placeholder: st.delta_generator.DeltaGenerator,
    output_dir: Path,
    project_context: str,
    target_fields: list[str],
    preferred_location: str,
    remote_ok: bool,
    max_results: int,
) -> Path | None:
    settings = load_settings()
    status_box = workflow_placeholder.empty()
    try:
        status_box.info("Running MCP provider discovery...")
        profiles, audit = discover_providers_via_mcp(
            project_context=project_context,
            target_fields=target_fields,
            preferred_location=preferred_location,
            remote_ok=remote_ok,
            max_results=max_results,
            settings=settings,
        )
    except Exception as exc:
        status_box.error(f"MCP discovery failed: {exc}")
        return None

    if not profiles:
        status_box.error("MCP discovery completed but returned no providers.")
        return None

    providers_path = write_discovery_output(
        output_path=output_dir / "discovered_providers.json",
        profiles=profiles,
        audit=audit,
    )
    status_box.success(
        f"MCP discovery found {len(profiles)} providers. Proceeding to ranking workflow."
    )
    with workflow_placeholder.expander("Discovered Providers", expanded=False):
        st.write(audit.get("search_summary") or "No search summary returned.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Name": profile.name,
                        "Type": profile.type,
                        "Expertise": profile.expertise,
                        "Location": profile.location,
                    }
                    for profile in profiles
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    return providers_path


def _preview_example_inputs() -> None:
    provider_preview, criteria_preview, brief_preview = st.tabs(
        ["Providers", "Criteria", "Project Brief"]
    )
    with provider_preview:
        st.dataframe(pd.read_csv(EXAMPLE_PROVIDERS), use_container_width=True)
    with criteria_preview:
        st.code(EXAMPLE_CRITERIA.read_text(encoding="utf-8"), language="yaml")
    with brief_preview:
        st.code(EXAMPLE_BRIEF.read_text(encoding="utf-8"), language="markdown")


def _preview_uploaded_inputs(
    uploaded_providers: UploadedFile | None,
    uploaded_criteria: UploadedFile | None,
    uploaded_brief: UploadedFile | None,
) -> None:
    tabs = st.tabs(["Providers", "Criteria", "Project Brief"])
    if uploaded_providers:
        with tabs[0]:
            if uploaded_providers.name.endswith(".csv"):
                st.dataframe(pd.read_csv(uploaded_providers), use_container_width=True)
            else:
                st.json(json.loads(uploaded_providers.getvalue().decode("utf-8")))
    else:
        with tabs[0]:
            st.caption("Upload a providers file to preview it.")

    if uploaded_criteria:
        with tabs[1]:
            st.code(uploaded_criteria.getvalue().decode("utf-8"), language="yaml")
    else:
        with tabs[1]:
            st.caption("Upload a criteria file to preview it.")

    if uploaded_brief:
        with tabs[2]:
            st.code(uploaded_brief.getvalue().decode("utf-8"), language="markdown")
    else:
        with tabs[2]:
            st.caption("Upload a brief file to preview it.")


def _preview_discovery_inputs(
    *,
    use_examples: bool,
    uploaded_criteria: UploadedFile | None,
    uploaded_brief: UploadedFile | None,
    project_context: str,
    target_fields: str,
    preferred_location: str,
    remote_ok: bool,
    max_results: int,
) -> None:
    st.info("Providers will be sourced live through the MCP bridge for this run.")
    tabs = st.tabs(["Discovery Request", "Criteria", "Project Brief"])
    with tabs[0]:
        st.json(
            {
                "project_context": project_context,
                "target_fields": _parse_target_fields(target_fields),
                "preferred_location": preferred_location,
                "remote_ok": remote_ok,
                "max_results": max_results,
            }
        )
    if use_examples:
        with tabs[1]:
            st.code(EXAMPLE_CRITERIA.read_text(encoding="utf-8"), language="yaml")
        with tabs[2]:
            st.code(EXAMPLE_BRIEF.read_text(encoding="utf-8"), language="markdown")
        return

    if uploaded_criteria:
        with tabs[1]:
            st.code(uploaded_criteria.getvalue().decode("utf-8"), language="yaml")
    else:
        with tabs[1]:
            st.caption("Upload a criteria file to preview it.")

    if uploaded_brief:
        with tabs[2]:
            st.code(uploaded_brief.getvalue().decode("utf-8"), language="markdown")
    else:
        with tabs[2]:
            st.caption("Upload a brief file to preview it.")


def _parse_target_fields(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            --light-surface: rgba(255, 255, 255, 0.96);
            --light-text: #000000;
            --dark-surface: #111827;
            --dark-surface-soft: #1f2937;
            --dark-text: #ffffff;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 113, 255, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(0, 200, 140, 0.10), transparent 24%),
                linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
        }
        .stApp,
        .stApp > div,
        .stApp [data-testid="stAppViewContainer"],
        .stApp [data-testid="stAppViewContainer"] > div,
        .stApp [data-testid="stHeader"],
        .stApp [data-testid="stToolbar"] {
            color: var(--light-text) !important;
        }
        .stApp .main .block-container,
        .stApp .main .block-container p,
        .stApp .main .block-container li,
        .stApp .main .block-container label,
        .stApp .main .block-container span,
        .stApp .main .block-container div,
        .stApp .main .block-container h1,
        .stApp .main .block-container h2,
        .stApp .main .block-container h3,
        .stApp .main .block-container h4 {
            color: var(--light-text) !important;
        }
        .stApp .main .block-container h1,
        .stApp .main .block-container h2,
        .stApp .main .block-container h3,
        .stApp .main .block-container h4,
        .stApp .main .block-container h5,
        .stApp .main .block-container h6,
        .stApp .main .block-container .stMarkdown,
        .stApp .main .block-container .stMarkdown *,
        .stApp .main .block-container [data-testid="stMarkdownContainer"],
        .stApp .main .block-container [data-testid="stMarkdownContainer"] *,
        .stApp .main .block-container [data-testid="stHeading"],
        .stApp .main .block-container [data-testid="stHeading"] *,
        .stApp .main .block-container [data-testid="stCaptionContainer"],
        .stApp .main .block-container [data-testid="stCaptionContainer"] * {
            color: var(--light-text) !important;
        }
        section[data-testid="stSidebar"] {
            background: var(--dark-surface) !important;
        }
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: var(--dark-text) !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] [data-baseweb="input"],
        section[data-testid="stSidebar"] [data-baseweb="base-input"],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
            background: var(--dark-surface-soft) !important;
            color: var(--dark-text) !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"] {
            background: #ff4b4b !important;
            color: var(--dark-text) !important;
            border: 0 !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"],
        section[data-testid="stSidebar"] button[kind="tertiary"] {
            color: var(--dark-text) !important;
            border-color: rgba(255, 255, 255, 0.24) !important;
        }
        div[data-testid="stMetric"] {
            background: var(--light-surface);
            border: 1px solid rgba(12, 22, 44, 0.08);
            padding: 0.85rem 1rem;
            border-radius: 8px;
            color: var(--light-text) !important;
        }
        div[data-testid="stExpander"] {
            background: var(--light-surface);
            border: 1px solid rgba(12, 22, 44, 0.08);
            border-radius: 8px;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary *,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[role="tablist"] button,
        div[role="tablist"] button *,
        div[data-testid="stDataFrame"] * {
            color: var(--light-text) !important;
        }
        .stApp .main button[kind] {
            color: var(--light-text) !important;
        }
        .stApp .main [data-testid="stCodeBlock"],
        .stApp .main [data-testid="stCodeBlock"] *,
        .stApp .main [data-testid="stCode"] *,
        .stApp .main [data-testid="stJson"],
        .stApp .main [data-testid="stJson"] *,
        .stApp .main [data-testid="stCodeBlock"] pre,
        .stApp .main [data-testid="stCode"] pre,
        .stApp .main pre,
        .stApp .main code {
            background: var(--dark-surface) !important;
            color: var(--dark-text) !important;
        }
        .stApp .main div[data-testid="stAlert"] * {
            color: var(--light-text) !important;
        }
        .stApp .main div[data-baseweb="notification"],
        .stApp .main div[data-baseweb="notification"] *,
        .stApp .main div[role="alert"],
        .stApp .main div[role="alert"] * {
            color: var(--light-text) !important;
        }
        .stApp .main [data-testid="stDataFrame"],
        .stApp .main [data-testid="stTable"],
        .stApp .main [role="tablist"] button[aria-selected="true"] {
            background: var(--light-surface) !important;
            color: var(--light-text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
