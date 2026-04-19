from __future__ import annotations

from provider_selection_agent import mcp_bridge
from provider_selection_agent.config import Settings
from provider_selection_agent.mcp_bridge import (
    BOOTSTRAP_PROVIDER_URLS,
    GROQ_STRUCTURED_OUTPUT_MODELS,
    MAX_TOTAL_CONTEXT_CHARS,
    SEARCH_BUFFER_RESULTS,
    CandidateSite,
    DiscoveredProvider,
    _bootstrap_candidate_urls,
    _build_page_context,
    _candidate_models,
    _extract_duckduckgo_redirect_target,
    _extract_expertise_keywords,
    _extract_json_object,
    _extract_page_text,
    _heuristic_extract_provider,
    _is_valid_provider_candidate,
    _normalize_url,
    _parse_search_result_links,
    _search_provider_pages,
    _should_retry_with_smaller_context,
    _should_try_fallback,
    _should_try_plain_json_fallback,
    handle_bridge_request,
)


def _settings() -> Settings:
    return Settings(
        llm_provider="gemini",
        llm_api_key="test-key",
        llm_model="gemini-2.5-flash",
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        log_level="INFO",
        vector_db_path=".local/vector_store",
        enable_web_search=True,
        mcp_server_url="http://127.0.0.1:8000/enrich",
        mcp_enrich_fields=("price", "expertise", "location", "availability"),
        mcp_timeout_seconds=120,
    )


def test_parse_search_result_links_handles_duckduckgo_redirects() -> None:
    html = (
        '<a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fabout">'
        "Example</a>"
    )
    links = _parse_search_result_links(html)
    assert links == ["https://example.com/about"]


def test_normalize_url_strips_trailing_slash() -> None:
    assert _normalize_url("https://example.com/about/") == "https://example.com/about"


def test_handle_bridge_request_rejects_unknown_operation() -> None:
    try:
        handle_bridge_request({"operation": "unknown"}, _settings())
    except ValueError as exc:
        assert "Unsupported operation" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown operation")


def test_search_parser_keeps_external_links_only() -> None:
    html = (
        '<a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fagency.example%2Fservices">A</a>'
        '<a href="https://duckduckgo.com/about">B</a>'
    )
    links = _parse_search_result_links(html)
    assert links == ["https://agency.example/services"]


def test_extract_duckduckgo_redirect_target_supports_relative_links() -> None:
    link = "/l/?uddg=https%3A%2F%2Fagency.example%2Fabout"
    assert _extract_duckduckgo_redirect_target(link) == "https://agency.example/about"


def test_invalid_provider_candidate_rejects_search_engine_and_unknown_expertise() -> None:
    provider = DiscoveredProvider(
        name="DuckDuckGo",
        type="search engine",
        expertise=[],
        location="Paoli, PA",
        portfolio_summary="unknown",
        relevance_rationale=(
            "No explicit evidence of Full-Stack Web Development "
            "or Data Architecture expertise."
        ),
    )
    candidate = CandidateSite(
        canonical_url="https://duckduckgo.com",
        pages=[("https://duckduckgo.com/about", "about text")],
    )

    assert _is_valid_provider_candidate(
        provider,
        candidate,
        ["Full-Stack Web Development", "Data Architecture"],
        "EMEA",
    ) is False


def test_valid_provider_candidate_accepts_relevant_agency() -> None:
    provider = DiscoveredProvider(
        name="Example Labs",
        type="Agency",
        expertise=["Full-Stack Web Development", "Data Architecture"],
        location="Warsaw, Poland",
        portfolio_summary="Built marketplaces and recommendation systems for digital platforms.",
        relevance_rationale="Strong match for marketplace backend and search work.",
    )
    candidate = CandidateSite(
        canonical_url="https://examplelabs.dev",
        pages=[("https://examplelabs.dev/services", "services text")],
    )

    assert _is_valid_provider_candidate(
        provider,
        candidate,
        ["Full-Stack Web Development", "Data Architecture"],
        "EMEA",
    ) is True


def test_valid_provider_candidate_accepts_llm_rationale_without_exact_keyword_match() -> None:
    provider = DiscoveredProvider(
        name="Signal Works",
        type="Agency",
        expertise=[],
        location="Berlin, Germany",
        portfolio_summary="Builds custom digital platforms and intelligent product backends.",
        relevance_rationale=(
            "The agency is a strong fit because it delivers custom backend platforms, "
            "discovery features, and personalization capabilities for digital products."
        ),
    )
    candidate = CandidateSite(
        canonical_url="https://signalworks.example",
        pages=[("https://signalworks.example/services", "custom backend platforms and personalization systems")],
    )

    assert _is_valid_provider_candidate(
        provider,
        candidate,
        ["Full-Stack Web Development", "Data Architecture"],
        "EMEA",
    ) is True


def test_bootstrap_candidate_urls_available_for_emea() -> None:
    urls = _bootstrap_candidate_urls("EMEA")
    assert len(urls) >= 4
    assert BOOTSTRAP_PROVIDER_URLS[0] in urls


def test_search_provider_pages_caps_candidate_collection(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_bridge,
        "_search_query_urls",
        lambda _query: [f"https://agency{i}.example" for i in range(20)],
    )

    urls = _search_provider_pages(
        "Need backend architecture and recommendation systems",
        ["Full-Stack Web Development", "Data Architecture"],
        "EMEA",
        5,
    )

    assert len(urls) == max(5 + SEARCH_BUFFER_RESULTS, 5 * 4)


def test_extract_page_text_removes_script_style_nav_and_footer_noise() -> None:
    html = """
    <html>
      <head>
        <style>.hero { color: red; }</style>
        <script>window.bad = true;</script>
      </head>
      <body>
        <nav>Home Services Contact</nav>
        <main>
          <h1>Backend engineering for EdTech platforms</h1>
          <p>We build search and recommendation systems for learning products.</p>
        </main>
        <footer>Privacy Terms Careers</footer>
      </body>
    </html>
    """

    text = _extract_page_text(html)

    assert "Backend engineering for EdTech platforms" in text
    assert "search and recommendation systems" in text
    assert "window.bad" not in text
    assert "Home Services Contact" not in text
    assert "Privacy Terms Careers" not in text


def test_build_page_context_stays_within_limit_and_prefers_relevant_text() -> None:
    pages = [
        (
            "https://example.com/about",
            "We are an agency building full-stack backend systems for marketplaces. " * 80,
        ),
        (
            "https://example.com/pricing",
            "Pricing starts at $10,000. Search and recommendation systems included. " * 80,
        ),
        (
            "https://example.com/misc",
            "Random unrelated text. " * 200,
        ),
    ]

    context = _build_page_context(
        pages,
        target_fields=["Full-Stack Web Development", "Data Architecture"],
        project_context="Need backend architecture and recommendation system support",
    )

    assert len(context) <= MAX_TOTAL_CONTEXT_CHARS
    assert (
        "Pricing starts at $10,000" in context
        or "full-stack backend systems" in context
        or "recommendation systems included" in context
    )


def test_groq_bridge_fallback_models_are_supported() -> None:
    settings = Settings(
        llm_provider="groq",
        llm_api_key="test-key",
        llm_model="llama-3.3-70b-versatile",
        llm_base_url="https://api.groq.com/openai/v1",
        log_level="INFO",
        vector_db_path=".local/vector_store",
        enable_web_search=True,
        mcp_server_url="http://127.0.0.1:8000/enrich",
        mcp_enrich_fields=("price", "expertise", "location", "availability"),
        mcp_timeout_seconds=120,
    )

    # User's model is now always kept first, with Groq fallbacks appended.
    assert _candidate_models(settings) == [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    assert "llama-3.3-70b-versatile" not in GROQ_STRUCTURED_OUTPUT_MODELS


def test_unsupported_json_schema_errors_trigger_fallback() -> None:
    error = RuntimeError(
        "Error code: 400 - {'error': {'message': "
        "'This model does not support response format `json_schema`'}}"
    )
    assert _should_try_fallback(error) is True


def test_request_too_large_errors_trigger_smaller_context_retry() -> None:
    error = RuntimeError(
        "Error code: 413 - {'error': {'message': 'Request too large for model "
        "`openai/gpt-oss-20b` on tokens per minute'}}"
    )
    assert _should_retry_with_smaller_context(error) is True


def test_timeout_errors_trigger_smaller_context_retry() -> None:
    assert _should_retry_with_smaller_context(TimeoutError("timed out")) is True


def test_length_limit_errors_trigger_smaller_context_retry() -> None:
    error = RuntimeError(
        "openai/gpt-oss-20b: Could not parse response content as the length limit was reached"
    )
    assert _should_retry_with_smaller_context(error) is True


def test_json_validate_failures_trigger_plain_json_fallback() -> None:
    error = RuntimeError("Error code: 400 - Failed to validate JSON (json_validate_failed)")
    assert _should_try_plain_json_fallback(error) is True


def test_extract_json_object_reads_embedded_json() -> None:
    payload = _extract_json_object(
        'Here is the result {"name":"Example","type":"Agency","expertise":["Data Architecture"]}'
    )
    assert payload["name"] == "Example"
    assert payload["type"] == "Agency"


def test_extract_expertise_keywords_matches_relevant_capabilities() -> None:
    expertise = _extract_expertise_keywords(
        (
            "We build full-stack products, scalable backend architecture, "
            "recommendation engines, and data platforms."
        ),
        ["Full-Stack Web Development", "Data Architecture"],
    )
    assert "Full-Stack Web Development" in expertise
    assert "Recommendation Systems" in expertise
    assert "Data Architecture" in expertise


def test_heuristic_extract_provider_salvages_strong_candidate() -> None:
    provider = _heuristic_extract_provider(
        CandidateSite(
            canonical_url="https://www.netguru.com",
            pages=[
                (
                    "https://www.netguru.com/services",
                    (
                        "Netguru is a Poland-based agency offering full-stack web development, "
                        "backend architecture, data engineering, search, and "
                        "recommendation systems "
                        "for marketplace platforms."
                    ),
                )
            ],
        ),
        project_context="Need backend architecture and recommendation systems",
        target_fields=["Full-Stack Web Development", "Data Architecture"],
        preferred_location="EMEA",
    )

    assert provider is not None
    assert provider.name == "Netguru"
    assert "Full-Stack Web Development" in provider.expertise


def test_handle_discovery_skips_extraction_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_bridge,
        "_search_provider_pages",
        lambda *_args, **_kwargs: ["https://good.example", "https://bad.example"],
    )
    monkeypatch.setattr(
        mcp_bridge,
        "_collect_candidate_sites",
        lambda _urls: [
            CandidateSite("https://good.example", [("https://good.example", "good")]),
            CandidateSite("https://bad.example", [("https://bad.example", "bad")]),
        ],
    )

    def fake_extract(
        candidate: CandidateSite,
        *_args,
        **_kwargs,
    ) -> DiscoveredProvider:
        if "bad.example" in candidate.canonical_url:
            raise RuntimeError("Model returned empty content")
        return DiscoveredProvider(
            name="Good Example",
            type="Agency",
            expertise=["Full-Stack Web Development"],
            location="Warsaw, Poland",
            portfolio_summary="Built marketplace backends.",
        )

    monkeypatch.setattr(mcp_bridge, "_extract_provider", fake_extract)

    response = handle_bridge_request(
        {
            "operation": "discover_providers",
            "project_context": "Need backend architecture",
            "target_fields": ["Full-Stack Web Development"],
            "preferred_location": "EMEA",
            "max_results": 5,
        },
        _settings(),
    )

    assert len(response["providers"]) == 1
    assert "skipped 1 extraction failures" in response["search_summary"]
