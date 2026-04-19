from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr

from provider_selection_agent.config import Settings, load_settings
from provider_selection_agent.sourcing import execute_sourcing_run

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
GROQ_STRUCTURED_OUTPUT_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}
DIRECTORY_DOMAINS = {
    "duckduckgo.com",
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "yahoo.com",
    "search.yahoo.com",
    "upwork.com",
    "clutch.co",
    "goodfirms.co",
    "sortlist.com",
    "designrush.com",
    "extract.co",
    "topdevelopers.co",
    "manifest.com",
    "themanifest.com",
    "agencyspotter.com",
    "linkedin.com",
}
SCRAPE_PATHS = (
    "",
    "/about",
    "/services",
    "/solutions",
    "/expertise",
    "/pricing",
    "/work",
    "/case-studies",
)
MIN_PROVIDER_RESULTS = 5
SEARCH_BUFFER_RESULTS = 15
FETCH_TIMEOUT_SECONDS = 60
MAX_PAGES_PER_CANDIDATE = 3
MAX_PAGE_TEXT_CHARS = 2500
MAX_TOTAL_CONTEXT_CHARS = 7000
CONTEXT_RETRY_BUDGETS = (
    (3, 2500, 7000),
    (2, 1500, 3200),
    (2, 900, 1800),
    (1, 700, 900),
)
EXTRACTION_RESPONSE_BUDGET = 1200
BOOTSTRAP_PROVIDER_URLS = (
    "https://www.netguru.com",
    "https://yalantis.com",
    "https://neontri.com",
    "https://addepto.com",
    "https://www.scnsoft.com",
    "https://www.datarootlabs.com",
    "https://www.itmagination.com",
)
EXPERTISE_PATTERNS = (
    ("Full-Stack Web Development", ("full-stack", "full stack", "web development")),
    ("Backend Architecture", ("backend architecture", "scalable backend", "backend systems")),
    ("Data Architecture", ("data architecture", "data engineering", "data platform")),
    ("Search Systems", ("search", "discovery", "semantic search")),
    ("Recommendation Systems", ("recommendation", "recommender", "personalization")),
    ("Marketplace Development", ("marketplace", "two-sided platform", "platform development")),
)
LOCATION_PATTERNS = (
    "poland",
    "ukraine",
    "london",
    "warsaw",
    "berlin",
    "lisbon",
    "barcelona",
    "amsterdam",
    "romania",
    "bulgaria",
    "egypt",
    "uae",
    "dubai",
    "saudi arabia",
    "south africa",
    "nigeria",
    "morocco",
)


class DiscoveryEvidence(BaseModel):
    field: str
    source: str


class DiscoveredProvider(BaseModel):
    name: str
    type: str
    expertise: list[str] = Field(default_factory=list)
    location: str = "unknown"
    price: float | None = None
    currency: str = "unknown"
    portfolio_summary: str = "unknown"
    source_type: str = "provider_site"
    relevance_rationale: str = ""
    evidence: list[DiscoveryEvidence] = Field(default_factory=list)


@dataclass
class CandidateSite:
    canonical_url: str
    pages: list[tuple[str, str]]


app = FastAPI()


@app.post("/enrich")
@app.post("/")
async def handle_mcp_request(request: FastAPIRequest) -> dict[str, Any]:
    payload = await request.json()
    try:
        return handle_bridge_request(payload, load_settings())
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


def handle_bridge_request(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    operation = str(payload.get("operation", "")).strip()
    if operation == "discover_providers":
        context = str(payload.get("project_context", "")).strip()
        if not context:
            raise ValueError("project_context is required")
        fields = ", ".join(
            str(item).strip() for item in payload.get("target_fields", []) if str(item).strip()
        )
        location = str(payload.get("preferred_location", "Any")).strip() or "Any"
        max_results = int(payload.get("max_results", 5))
        max_results = max(1, min(max_results, 10))
        sourcing_results = execute_sourcing_run(context, fields, location, max_results)
        return {
            "providers": sourcing_results["providers"],
            "search_summary": (
                sourcing_results.get("search_summary")
                or f"Autonomous agent completed search for {fields or 'provider discovery'}."
            ),
        }
    raise ValueError("Unsupported operation. Expected 'discover_providers'.")


def handle_legacy_discovery(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    project_context = str(payload.get("project_context", "")).strip()
    if not project_context:
        raise ValueError("project_context is required")

    target_fields = [
        str(item).strip() for item in payload.get("target_fields", []) if str(item).strip()
    ]
    preferred_location = str(payload.get("preferred_location", "unknown")).strip()
    max_results = int(payload.get("max_results", 5))
    max_results = max(1, min(max_results, 10))

    search_results = _search_provider_pages(
        project_context,
        target_fields,
        preferred_location,
        max_results,
    )
    candidates = _collect_candidate_sites(search_results)

    discovered: list[dict[str, Any]] = []
    rejected_candidates = 0
    extraction_failures = 0
    logger.info(
        "Starting extraction for %d candidate sites (target: %d providers)",
        len(candidates), max_results,
    )
    for candidate in candidates:
        if len(discovered) >= max_results:
            break
        logger.info("Extracting provider from %s", candidate.canonical_url)
        try:
            provider = _extract_provider(
                candidate,
                project_context,
                target_fields,
                preferred_location,
                settings,
            )
        except Exception as exc:
            extraction_failures += 1
            logger.warning(
                "Extraction FAILED for %s: %s", candidate.canonical_url, exc,
            )
            continue
        if provider is None:
            rejected_candidates += 1
            logger.info("Rejected weak candidate: %s", candidate.canonical_url)
            continue
        logger.info("Accepted provider: %s from %s", provider.name, candidate.canonical_url)
        discovered.append(provider.model_dump(mode="json"))

    summary = (
        f"Collected {len(search_results)} candidate URLs across "
        f"{len(candidates)} provider domains; "
        f"returned {len(discovered)} normalized providers and rejected "
        f"{rejected_candidates} weak candidates and skipped "
        f"{extraction_failures} extraction failures."
    )
    logger.info("Discovery summary: %s", summary)
    return {"providers": discovered, "search_summary": summary}


def _search_provider_pages(
    project_context: str,
    target_fields: list[str],
    preferred_location: str,
    max_results: int,
) -> list[str]:
    queries = _build_queries(project_context, target_fields, preferred_location)
    discovered_urls: list[str] = []
    seen: set[str] = set()
    seen_domains: set[str] = set()
    target_candidate_count = max(
        MIN_PROVIDER_RESULTS,
        max_results + SEARCH_BUFFER_RESULTS,
        max_results * 4,
    )
    for query in queries:
        logger.info("Search query: %s", query)
        urls = _search_query_urls(query)
        logger.info("  => %d raw URLs returned", len(urls))
        for url in urls:
            domain = urlparse(url).netloc.lower()
            if not domain or any(domain.endswith(blocked) for blocked in DIRECTORY_DOMAINS):
                continue
            normalized = _normalize_url(url)
            if normalized in seen:
                continue
            seen.add(normalized)
            seen_domains.add(domain)
            discovered_urls.append(normalized)
            if len(seen_domains) >= target_candidate_count:
                logger.info(
                    "Reached %d unique domains, stopping search.", len(seen_domains),
                )
                return discovered_urls

    # Fallback: if we don't have enough UNIQUE DOMAINS, inject bootstrap URLs.
    # Previously this checked len(discovered_urls), which could be inflated by
    # multiple pages from one irrelevant domain.
    if len(seen_domains) < target_candidate_count:
        logger.info(
            "Only %d unique domains found, injecting bootstrap URLs.", len(seen_domains),
        )
        for url in _bootstrap_candidate_urls(preferred_location, target_candidate_count):
            normalized = _normalize_url(url)
            if normalized in seen:
                continue
            seen.add(normalized)
            seen_domains.add(urlparse(normalized).netloc.lower())
            discovered_urls.append(normalized)
            if len(seen_domains) >= target_candidate_count:
                break
    return discovered_urls


def _bootstrap_candidate_urls(preferred_location: str, limit: int | None = None) -> list[str]:
    preferred = preferred_location.lower()
    if preferred in {"emea", "europe", "middle east", "africa"}:
        urls = list(BOOTSTRAP_PROVIDER_URLS)
    else:
        urls = list(BOOTSTRAP_PROVIDER_URLS[:4])
    if limit is None:
        return urls
    return urls[:limit]


def _search_query_urls(query: str) -> list[str]:
    try:
        from duckduckgo_search import DDGS
        results: list[str] = []
        seen: set[str] = set()
        with DDGS() as ddgs:
            for r in ddgs.text(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                max_results=25,
            ):
                url = ""
                if isinstance(r, dict):
                    # duckduckgo-search returns the URL under "link" in current versions.
                    url = (
                        r.get("link")
                        or r.get("href")
                        or r.get("url")
                        or ""
                    )
                normalized = _normalize_url(url) if url else ""
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    results.append(normalized)
        return results
    except ImportError:
        raise ImportError(
            "Please install duckduckgo-search to enable reliable free web search. "
            "Run: pip install duckduckgo-search"
        )
    except Exception as exc:
        logger.warning("Search failed for query %r: %s", query, exc)
        # Failsafe if service is unavailable for this query.
        return []


# Map broad regional terms to real location keywords agencies actually use.
_REGION_EXPANSIONS: dict[str, list[str]] = {
    "emea": ["Europe", "Poland", "UK", "Germany", "Romania", "Ukraine"],
    "europe": ["Poland", "UK", "Germany", "Romania", "Netherlands"],
    "middle east": ["UAE", "Dubai", "Saudi Arabia"],
    "africa": ["South Africa", "Nigeria", "Egypt", "Morocco"],
    "apac": ["India", "Singapore", "Australia"],
    "latam": ["Brazil", "Argentina", "Mexico", "Colombia"],
    "north america": ["USA", "Canada"],
}


def _build_queries(
    project_context: str, target_fields: list[str], preferred_location: str
) -> list[str]:
    target_text = " ".join(target_fields) if target_fields else "software development"
    compact_context = " ".join(project_context.split()[:24])

    # Expand broad regional labels into concrete locations for search.
    location_key = preferred_location.strip().lower()
    location_variants = _REGION_EXPANSIONS.get(location_key, [preferred_location])

    queries: list[str] = []

    # Location-specific queries (use the first 3 expanded locations)
    for loc in location_variants[:3]:
        queries.append(
            f'backend architecture recommendation system agency {loc}'
        )
        queries.append(
            f'edtech platform development agency search recommendation {loc}'
        )

    # Location-agnostic queries (critical safety net)
    queries.extend([
        f'{target_text} software development agency services portfolio',
        'EdTech platform backend development agency tutoring courses',
        'search recommendation system software development company',
        'custom software agency backend API architecture services',
        'llm search recommendation backend engineering agency',
        'marketplace backend architecture data platform development agency',
    ])

    # Project-context query as a last resort
    queries.append(compact_context)

    deduped_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split()).strip()
        if normalized and normalized not in seen_queries:
            seen_queries.add(normalized)
            deduped_queries.append(normalized)
    return deduped_queries


def _parse_search_result_links(html: str) -> list[str]:
    links = re.findall(r'href="([^"]+)"', html)
    results: list[str] = []
    for link in links:
        redirect_target = _extract_duckduckgo_redirect_target(link)
        if redirect_target:
            results.append(redirect_target)
        elif link.startswith("http"):
            parsed = urlparse(link)
            if (
                parsed.scheme in {"http", "https"}
                and parsed.netloc
                and not parsed.netloc.lower().endswith("duckduckgo.com")
            ):
                results.append(link)
    return results


def _extract_duckduckgo_redirect_target(link: str) -> str | None:
    if not (
        "uddg=" in link
        and (
            "duckduckgo.com/l/?" in link
            or link.startswith("/l/?")
            or link.startswith("//duckduckgo.com/l/?")
        )
    ):
        return None

    resolved = urljoin("https://duckduckgo.com", link)
    query = parse_qs(urlparse(resolved).query)
    target = query.get("uddg", [""])[0]
    if not target:
        return None
    clean_target = unquote(target)
    parsed = urlparse(clean_target)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        if parsed.netloc.lower().endswith("duckduckgo.com"):
            return None
        return clean_target
    return None


def _collect_candidate_sites(urls: list[str]) -> list[CandidateSite]:
    by_domain: dict[str, list[str]] = {}
    for url in urls:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain:
            continue
        by_domain.setdefault(domain, []).append(f"{parsed.scheme}://{parsed.netloc}")

    candidates: list[CandidateSite] = []
    for _domain, roots in by_domain.items():
        root = roots[0]
        pages: list[tuple[str, str]] = []
        for path in SCRAPE_PATHS:
            candidate_url = urljoin(root, path)
            try:
                text = _extract_page_text(_fetch_url(candidate_url))
            except Exception:
                continue
            if not text:
                continue
            pages.append((candidate_url, text[:MAX_PAGE_TEXT_CHARS]))
            if len(pages) >= MAX_PAGES_PER_CANDIDATE:
                break
        if pages:
            candidates.append(CandidateSite(canonical_url=root, pages=pages))
    return candidates


def _extract_provider(
    candidate: CandidateSite,
    project_context: str,
    target_fields: list[str],
    preferred_location: str,
    settings: Settings,
) -> DiscoveredProvider | None:
    if not settings.llm_api_key:
        raise ValueError(f"{settings.api_key_env_hint} is not configured for MCP bridge extraction")

    extraction_errors: list[str] = []
    for max_pages, max_page_chars, max_total_chars in CONTEXT_RETRY_BUDGETS:
        page_context = _build_page_context(
            candidate.pages,
            target_fields=target_fields,
            project_context=project_context,
            max_pages=max_pages,
            max_page_chars=max_page_chars,
            max_total_chars=max_total_chars,
        )
        prompt = (
            "You are an autonomous sourcing extraction agent. "
            "Extract exactly one provider profile from the scraped provider website pages. "
            "Use only explicit information from the provided pages. "
            "If price or location is not explicit, return null or 'unknown'. "
            "Do not infer from directories or third-party marketplaces. "
            "Return expertise as a short list of explicit service capabilities. "
            "Be concise. Return only the structured result with no commentary.\n\n"
            f"Project context:\n{project_context}\n\n"
            f"Target fields: {target_fields}\n"
            f"Preferred location: {preferred_location}\n\n"
            f"Scraped pages:\n{page_context}\n"
        )
        try:
            result = _invoke_extractor_model(prompt, settings)
        except Exception as exc:
            extraction_errors.append(str(exc))
            logger.debug(
                "Extractor model error for %s (budget %d/%d): %s",
                candidate.canonical_url, max_page_chars, max_total_chars, exc,
            )
            # Always try the next (smaller) context budget before giving up.
            continue
        provider = (
            result
            if isinstance(result, DiscoveredProvider)
            else DiscoveredProvider.model_validate(result)
        )
        if not _is_valid_provider_candidate(provider, candidate, target_fields, preferred_location):
            return None
        if not provider.evidence:
            provider.evidence = [
                DiscoveryEvidence(field="source", source=url) for url, _ in candidate.pages[:3]
            ]
        return provider

    heuristic_provider = _heuristic_extract_provider(
        candidate,
        project_context=project_context,
        target_fields=target_fields,
        preferred_location=preferred_location,
    )
    if heuristic_provider is not None:
        return heuristic_provider
    if extraction_errors:
        raise RuntimeError(extraction_errors[-1])
    return None


def _build_page_context(
    pages: list[tuple[str, str]],
    *,
    target_fields: list[str],
    project_context: str,
    max_pages: int = MAX_PAGES_PER_CANDIDATE,
    max_page_chars: int = MAX_PAGE_TEXT_CHARS,
    max_total_chars: int = MAX_TOTAL_CONTEXT_CHARS,
) -> str:
    keywords = {
        token.lower()
        for token in (
            "agency",
            "freelancer",
            "full-stack",
            "backend",
            "data",
            "architecture",
            "search",
            "recommendation",
            "marketplace",
            "pricing",
            "services",
            "about",
            *target_fields,
            *project_context.split(),
        )
        if token
    }

    ranked_pages = sorted(
        pages,
        key=lambda item: _page_relevance_score(item[1], keywords),
        reverse=True,
    )

    chunks: list[str] = []
    total_chars = 0
    for url, text in ranked_pages[:max_pages]:
        snippet = _best_relevant_snippet(text, keywords, max_page_chars)
        chunk = f"URL: {url}\nCONTENT:\n{snippet}"
        if total_chars + len(chunk) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                break
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total_chars += len(chunk)
        if total_chars >= max_total_chars:
            break
    return "\n\n".join(chunks)


def _page_relevance_score(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword and keyword in lowered)


def _best_relevant_snippet(text: str, keywords: set[str], limit: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored_sentences = sorted(
        sentences,
        key=lambda sentence: _page_relevance_score(sentence, keywords),
        reverse=True,
    )
    selected: list[str] = []
    total = 0
    for sentence in scored_sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue
        if total + len(cleaned) + 1 > limit:
            continue
        selected.append(cleaned)
        total += len(cleaned) + 1
        if total >= limit or len(selected) >= 12:
            break
    if not selected:
        return text[:limit]
    return " ".join(selected)[:limit]


def _heuristic_extract_provider(
    candidate: CandidateSite,
    *,
    project_context: str,
    target_fields: list[str],
    preferred_location: str,
) -> DiscoveredProvider | None:
    combined_text = " ".join(text for _, text in candidate.pages)
    expertise = _extract_expertise_keywords(combined_text, target_fields)
    if not expertise:
        return None

    portfolio_summary = _extract_portfolio_summary(combined_text, target_fields, project_context)
    provider = DiscoveredProvider(
        name=_derive_provider_name(candidate),
        type="Agency",
        expertise=expertise,
        location=_extract_location(combined_text),
        price=_extract_price(combined_text),
        currency=_extract_currency(combined_text),
        portfolio_summary=portfolio_summary,
        source_type="provider_site_heuristic",
        relevance_rationale=(
            "Heuristic extraction from provider site content; matched target capabilities in "
            f"{', '.join(expertise)}."
        ),
        evidence=[DiscoveryEvidence(field="source", source=url) for url, _ in candidate.pages[:3]],
    )
    if _is_valid_provider_candidate(provider, candidate, target_fields, preferred_location):
        return provider
    return None


def _derive_provider_name(candidate: CandidateSite) -> str:
    host = urlparse(candidate.canonical_url).netloc.lower()
    host = host.removeprefix("www.")
    label = host.split(".", 1)[0]
    if label == "scnsoft":
        return "ScienceSoft"
    if label == "itmagination":
        return "ITMAGINATION"
    parts = re.findall(r"[A-Za-z][a-z]*|[A-Z]+(?![a-z])|\d+", label)
    if parts:
        return " ".join(part.capitalize() for part in parts)
    return label.capitalize() or "Unknown"


def _extract_expertise_keywords(text: str, target_fields: list[str]) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for label, phrases in EXPERTISE_PATTERNS:
        if any(phrase in lowered for phrase in phrases):
            matched.append(label)
    for field in target_fields:
        normalized = field.strip()
        if normalized and normalized not in matched:
            field_lower = normalized.lower()
            if any(token in lowered for token in field_lower.split()):
                matched.append(normalized)
    return matched[:5]


def _extract_portfolio_summary(
    text: str,
    target_fields: list[str],
    project_context: str,
) -> str:
    keywords = {
        token.lower()
        for token in (*target_fields, *project_context.split(), "platform", "build", "services")
        if token
    }
    snippet = _best_relevant_snippet(text, keywords, 320).strip()
    if not snippet:
        return "unknown"
    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    summary = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip()).strip()
    return summary or "unknown"


def _extract_location(text: str) -> str:
    lowered = text.lower()
    for location in LOCATION_PATTERNS:
        if location in lowered:
            return location.title()
    return "unknown"


def _extract_price(text: str) -> float | None:
    match = re.search(r"(?:\$|usd|eur|gbp)\s?(\d{2,6}(?:[.,]\d{1,2})?)", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _extract_currency(text: str) -> str:
    lowered = text.lower()
    if "$" in text or "usd" in lowered:
        return "USD"
    if "eur" in lowered or "€" in text:
        return "EUR"
    if "gbp" in lowered or "£" in text:
        return "GBP"
    return "unknown"


def _invoke_extractor_model(prompt: str, settings: Settings) -> DiscoveredProvider:
    errors: list[str] = []
    for model_name in _candidate_models(settings):
        # --- Attempt 1: Structured output (JSON schema) ---
        try:
            return _call_structured_provider_model(prompt, settings, model_name)
        except Exception as exc:
            errors.append(f"{model_name}[structured]: {exc}")
            logger.debug("Structured extraction failed with %s: %s", model_name, exc)

        # --- Attempt 2: Plain JSON prompt (always tried as fallback) ---
        try:
            return _call_plain_json_provider_model(prompt, settings, model_name)
        except Exception as fallback_exc:
            errors.append(f"{model_name}[plain_json]: {fallback_exc}")
            logger.debug("Plain JSON extraction failed with %s: %s", model_name, fallback_exc)
            if not _should_try_fallback(fallback_exc):
                break

    raise RuntimeError(
        "Provider extraction failed for all candidate models. "
        + " | ".join(errors)
    )


def _candidate_models(settings: Settings) -> list[str]:
    models = [settings.llm_model]
    if settings.llm_provider == "gemini" and settings.llm_model != "gemini-2.5-flash":
        models.append("gemini-2.5-flash")
    if settings.llm_provider == "groq":
        # Always keep the user's configured model first — even if it doesn't
        # support structured output, _invoke_extractor_model will fall back to
        # plain JSON mode automatically.
        groq_fallbacks = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
        for fb in groq_fallbacks:
            if fb not in models:
                models.append(fb)
    return list(dict.fromkeys(models))


def _call_structured_provider_model(
    prompt: str,
    settings: Settings,
    model_name: str,
) -> DiscoveredProvider:
    from langchain_openai import ChatOpenAI

    if settings.llm_base_url:
        llm = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(settings.llm_api_key),
            temperature=0,
            base_url=settings.llm_base_url,
            max_tokens=EXTRACTION_RESPONSE_BUDGET,
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(settings.llm_api_key),
            temperature=0,
            max_tokens=EXTRACTION_RESPONSE_BUDGET,
        )
    result = llm.with_structured_output(DiscoveredProvider).invoke(prompt)
    return (
        result
        if isinstance(result, DiscoveredProvider)
        else DiscoveredProvider.model_validate(result)
    )


def _call_plain_json_provider_model(
    prompt: str,
    settings: Settings,
    model_name: str,
) -> DiscoveredProvider:
    from langchain_openai import ChatOpenAI

    json_prompt = (
        f"{prompt}\n\n"
        "Return only a single compact JSON object with these keys exactly:\n"
        'name, type, expertise, location, price, currency, portfolio_summary, '
        "source_type, relevance_rationale, evidence.\n"
        "Rules:\n"
        "- expertise must be an array of strings.\n"
        "- evidence must be an array of objects with keys field and source.\n"
        '- use null for unknown numeric values and "unknown" for unknown strings.\n'
        "- no markdown, no explanation, no code fences.\n"
    )
    llm_kwargs = {
        "model": model_name,
        "api_key": SecretStr(settings.llm_api_key),
        "temperature": 0,
        "max_tokens": EXTRACTION_RESPONSE_BUDGET,
    }
    if settings.llm_base_url:
        llm_kwargs["base_url"] = settings.llm_base_url
    llm = ChatOpenAI(**llm_kwargs)
    response = llm.invoke(json_prompt)
    content = response.content if hasattr(response, "content") else response
    return DiscoveredProvider.model_validate(_extract_json_object(content))


def _should_try_fallback(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in (
            "quota exceeded",
            "resource_exhausted",
            "rate limit",
            "429",
            "does not support response format",
            "json_schema",
            "invalid_request_error",
            "json_validate_failed",
            "empty content",
            "model not found",
            "model_not_found",
            "not available",
            "does not exist",
            "connection error",
            "timed out",
            "timeout",
        )
    )


def _should_retry_with_smaller_context(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in (
            "413",
            "request too large",
            "tokens per minute",
            "rate_limit_exceeded",
            "length limit was reached",
            "could not parse response content as the length limit was reached",
            "timed out",
            "timeout",
        )
    )


def _should_try_plain_json_fallback(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in (
            "json_validate_failed",
            "failed to validate json",
            "could not parse response content",
        )
    )


def _extract_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    text = str(content).strip()
    if not text:
        raise ValueError("Model returned empty content")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return a JSON object")
    return json.loads(match.group(0))


def _is_valid_provider_candidate(
    provider: DiscoveredProvider,
    candidate: CandidateSite,
    target_fields: list[str],
    preferred_location: str,
) -> bool:
    if not provider.name or provider.name.lower() == "unknown":
        return False

    parsed = urlparse(candidate.canonical_url)
    domain = parsed.netloc.lower()

    if not domain or any(domain.endswith(blocked) for blocked in DIRECTORY_DOMAINS):
        return False

    provider_type = provider.type.strip().lower()
    allowed_provider_type_tokens = (
        "agency",
        "company",
        "studio",
        "consultancy",
        "firm",
        "partner",
    )
    blocked_provider_type_tokens = (
        "directory",
        "marketplace",
        "search engine",
        "job board",
        "listing",
        "aggregator",
        "platform",
    )
    if any(token in provider_type for token in blocked_provider_type_tokens):
        return False
    if provider_type and not any(token in provider_type for token in allowed_provider_type_tokens):
        return False

    rationale = provider.relevance_rationale.strip()
    if not provider.expertise and not rationale:
        return False

    negative_rationale_markers = (
        "not relevant",
        "no evidence",
        "insufficient evidence",
        "directory listing",
        "marketplace listing",
        "aggregator",
        "search engine",
        "job board",
        "not an agency",
    )
    rationale_lower = rationale.lower()
    if rationale and any(marker in rationale_lower for marker in negative_rationale_markers):
        return False

    text_signal = " ".join(
        part
        for part in (
            provider.portfolio_summary,
            rationale,
            " ".join(provider.expertise),
            " ".join(text for _, text in candidate.pages),
        )
        if part and part != "unknown"
    ).lower()
    if not any(
        token in text_signal
        for token in (
            "backend",
            "api",
            "platform",
            "architecture",
            "data",
            "search",
            "recommend",
            "personalization",
            "marketplace",
            "edtech",
            "software development",
        )
    ):
        return False

    return True


def _fetch_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def _extract_page_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Strip noisy elements so the LLM sees mostly human-readable site copy.
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()

        text_fragments = [
            fragment.strip()
            for fragment in soup.stripped_strings
            if fragment and len(fragment.strip()) > 1
        ]
        text = " ".join(text_fragments)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:20000]
    except ImportError:
        raise ImportError(
            "Please install beautifulsoup4 for text extraction. "
            "Run: pip install beautifulsoup4"
        )


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
