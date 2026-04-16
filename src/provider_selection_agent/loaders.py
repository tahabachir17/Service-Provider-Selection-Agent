from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from provider_selection_agent.models import CriteriaConfig, ProviderProfile


def load_providers(path: str | Path) -> list[ProviderProfile]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"provider file not found: {source}")

    if source.suffix.lower() == ".csv":
        rows = pd.read_csv(source).fillna("").to_dict(orient="records")
    elif source.suffix.lower() == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("providers", [])
    else:
        raise ValueError("providers must be a CSV or JSON file")

    if not isinstance(rows, list):
        raise ValueError("provider file must contain a list of provider records")

    providers = [ProviderProfile.model_validate(_normalize_unknowns(row)) for row in rows]
    _reject_duplicate_names(providers)
    return providers


def load_criteria(path: str | Path) -> CriteriaConfig:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"criteria file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return CriteriaConfig.model_validate(data)


def load_brief(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"project brief not found: {source}")
    return source.read_text(encoding="utf-8").strip()


def _normalize_unknowns(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            normalized[key] = "unknown"
        elif isinstance(value, str) and not value.strip():
            normalized[key] = "unknown"
        else:
            normalized[key] = value
    return normalized


def _reject_duplicate_names(providers: list[ProviderProfile]) -> None:
    names = [provider.name.lower() for provider in providers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate provider names found: {duplicates}")
