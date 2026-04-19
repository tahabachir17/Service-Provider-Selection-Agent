from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from provider_selection_agent.config import load_settings
from provider_selection_agent.sourcing import discover_providers_via_mcp, write_discovery_output
from provider_selection_agent.workflow import run_workflow

app = typer.Typer(help="Compare and rank service providers.")
console = Console()


@app.command()
def compare(
    providers: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="CSV or JSON provider file."),
    ],
    criteria: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="YAML criteria config."),
    ],
    brief: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="Markdown project brief."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for recommendation artifacts."),
    ],
) -> None:
    """Run a provider comparison and write draft recommendation artifacts."""
    settings = load_settings()
    state = run_workflow(
        providers_path=str(providers),
        criteria_path=str(criteria),
        brief_path=str(brief),
        output_dir=str(out),
        settings=settings,
    )
    winner = state.report.synthesis.recommended_provider if state.report else None
    console.print("[green]Comparison complete.[/green] Draft status: DRAFT_PENDING_APPROVAL")
    console.print(f"Recommended provider: {winner or 'No eligible provider'}")
    console.print(f"Artifacts written to: {out}")


@app.command()
def discover(
    context: Annotated[
        str,
        typer.Option(help="Project context used to source providers via the MCP bridge."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output JSON file for discovered providers."),
    ],
    target_fields: Annotated[
        list[str],
        typer.Option(help="Target capability fields to search for.", rich_help_panel="Discovery"),
    ] = ("Full-Stack Web Development", "Data Architecture"),
    preferred_location: Annotated[
        str,
        typer.Option(help="Preferred provider geography."),
    ] = "EMEA",
    remote_ok: Annotated[
        bool,
        typer.Option(help="Allow remote providers outside the preferred location."),
    ] = True,
    max_results: Annotated[
        int,
        typer.Option(min=1, max=10, help="Maximum number of providers to write."),
    ] = 5,
) -> None:
    """Discover providers via MCP search/scrape and write a providers JSON file."""
    settings = load_settings()
    profiles, audit = discover_providers_via_mcp(
        project_context=context,
        target_fields=target_fields,
        preferred_location=preferred_location,
        remote_ok=remote_ok,
        max_results=max_results,
        settings=settings,
    )
    output_path = write_discovery_output(output_path=out, profiles=profiles, audit=audit)
    console.print(f"[green]Discovery complete.[/green] Providers found: {len(profiles)}")
    console.print(f"Providers written to: {output_path}")


if __name__ == "__main__":
    app()
