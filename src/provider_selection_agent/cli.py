from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from provider_selection_agent.config import load_settings
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


if __name__ == "__main__":
    app()
