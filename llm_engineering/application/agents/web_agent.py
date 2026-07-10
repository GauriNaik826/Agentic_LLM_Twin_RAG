# What: Postpone runtime evaluation of type hints.
# Why: Keeps forward-reference typing safe and lightweight at import time.
from __future__ import annotations

# What: Import JSON serialization/deserialization helpers.
# Why: Needed to build request payloads and parse provider responses.
import json
# What: Import environment variable access.
# Why: API keys for Tavily/Serper are read from environment.
import os
# What: Import socket timeout type.
# Why: Distinguish network timeout from other URL errors.
import socket
# What: Import date/time helpers.
# Why: Used to reject stale web results by age.
from datetime import date, datetime, timedelta
# What: Import generic typing alias.
# Why: Search provider payloads are dynamic dictionaries.
from typing import Any
# What: Import URL parser.
# Why: Normalize domains for trusted-source checks.
from urllib.parse import urlparse
# What: Import low-level HTTP request utilities.
# Why: Calls external web search providers without extra dependencies.
from urllib import request
# What: Import URL error class.
# Why: Handle timeout and request failures explicitly.
from urllib.error import URLError

# What: Import Opik instrumentation.
# Why: Adds tracing to web-agent execution paths.
import opik
# What: Import prompt template builder.
# Why: Structures summarization request for LLM.
from langchain_core.prompts import ChatPromptTemplate
# What: Import OpenAI chat wrapper.
# Why: Used to summarize validated search results.
from langchain_openai import ChatOpenAI

# What: Import circuit breaker primitives.
# Why: Protect search dependencies from repeated failures.
from llm_engineering.application.guardrails import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
# What: Import app settings.
# Why: Provides OpenAI model ID and API key.
from llm_engineering.settings import settings


# What: Custom timeout error for web search layer.
# Why: Lets supervisor apply precise fallback policies for timeouts.
class WebSearchTimeoutError(TimeoutError):
    """Raised when web search times out."""


# What: Web agent encapsulating Search -> Extract -> Validate -> Summarize.
# Why: Keeps web retrieval logic modular and reusable by supervisor.
class WebAgent:
    """Web search agent with Search -> Extract -> Summarize workflow."""

    # What: Allowlist of trusted source domains.
    # Why: Filters low-quality or untrusted websites.
    TRUSTED_DOMAINS = {
        "reuters.com",
        "bbc.com",
        "openai.com",
        "anthropic.com",
    }
    # What: Maximum acceptable content age in days.
    # Why: Prevent stale web content in responses.
    MAX_RESULT_AGE_DAYS = 365
    # What: Minimum content size for a result.
    # Why: Rejects empty/snippet-only weak sources.
    MIN_CONTENT_LENGTH = 20
    # What: Circuit breaker protecting web search provider calls.
    # Why: Opens after repeated failures and recovers via half-open probes.
    _search_cb = CircuitBreaker(
        name="web_search",
        config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30, half_open_max_calls=1),
    )

    # What: Execute Tavily search.
    # Why: Primary provider in auto mode.
    @staticmethod
    def _search_tavily(query: str, k: int = 5) -> list[dict[str, Any]]:
        # What: Read Tavily key from environment.
        # Why: Avoid hardcoding secrets in code.
        api_key = os.getenv("TAVILY_API_KEY")
        # What: Return no results when key is absent.
        # Why: Caller can fallback to alternate provider.
        if not api_key:
            return []

        # What: Build provider request payload.
        # Why: Encodes query and search depth options.
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": k,
            "search_depth": "advanced",
        }
        # What: Construct HTTP POST request.
        # Why: Tavily endpoint expects JSON body.
        req = request.Request(
            url="https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # What: Execute request and parse JSON response.
        # Why: Get raw provider result list.
        try:
            with request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        # What: Map low-level timeout to domain-specific timeout error.
        # Why: Enables explicit timeout fallback in supervisor.
        except URLError as exc:
            if isinstance(getattr(exc, "reason", None), socket.timeout):
                raise WebSearchTimeoutError("Tavily search timed out") from exc
            # What: Re-raise non-timeout failures unchanged.
            # Why: Preserve original failure semantics.
            raise

        # What: Return provider results list.
        # Why: Downstream extract/validation pipeline expects list payload.
        return data.get("results", [])

    # What: Execute Serper search.
    # Why: Secondary provider and fallback when Tavily is unavailable.
    @staticmethod
    def _search_serper(query: str, k: int = 5) -> list[dict[str, Any]]:
        # What: Read Serper key from environment.
        # Why: Avoid secret leakage and keep config external.
        api_key = os.getenv("SERPER_API_KEY")
        # What: Return empty results when key is absent.
        # Why: Allows calling flow to decide fallback policy.
        if not api_key:
            return []

        # What: Build Serper request payload.
        # Why: Sets query and desired number of results.
        payload = {"q": query, "num": k}
        # What: Build HTTP POST request.
        # Why: Serper API expects JSON with API key header.
        req = request.Request(
            url="https://google.serper.dev/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
            method="POST",
        )

        # What: Execute request and decode JSON response.
        # Why: Retrieve organic web results from Serper.
        try:
            with request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        # What: Convert socket timeout into domain timeout error.
        # Why: Keeps timeout handling consistent across providers.
        except URLError as exc:
            if isinstance(getattr(exc, "reason", None), socket.timeout):
                raise WebSearchTimeoutError("Serper search timed out") from exc
            # What: Re-raise non-timeout errors.
            # Why: Preserve failure details for circuit breaker and caller.
            raise

        # What: Return organic results array.
        # Why: This is Serper's primary result collection.
        return data.get("organic", [])

    # What: Normalize heterogeneous provider fields into a common schema.
    # Why: Downstream validation/summarization should be provider-agnostic.
    @staticmethod
    def _extract(search_results: list[dict[str, Any]]) -> list[dict[str, str]]:
        # What: Output list for normalized results.
        # Why: Preserve extracted subset in stable structure.
        extracted: list[dict[str, str]] = []

        # What: Iterate over raw provider result items.
        # Why: Extract title/url/content/date with tolerant field names.
        for result in search_results:
            title = str(result.get("title", "")).strip()
            url = str(result.get("url", result.get("link", ""))).strip()
            content = str(result.get("content", result.get("snippet", ""))).strip()

            # What: Keep only non-empty entries.
            # Why: Skip blank payload rows.
            if title or url or content:
                extracted.append(
                    {
                        # What: Normalized title field.
                        # Why: Used in output summaries and diagnostics.
                        "title": title,
                        # What: Normalized URL field.
                        # Why: Used for trust checks and citations.
                        "url": url,
                        # What: Normalized content/snippet field.
                        # Why: Primary evidence text for summarization.
                        "content": content,
                        # What: Best-effort publication date field.
                        # Why: Used by freshness guardrail.
                        "published_date": str(
                            result.get("published_date")
                            or result.get("date")
                            or result.get("published")
                            or ""
                        ).strip(),
                    }
                )

        # What: Return normalized result list.
        # Why: Input for validation stage.
        return extracted

    # What: Normalize URL into canonical domain form.
    # Why: Domain allowlist checks should not fail on casing/www variants.
    @staticmethod
    def _normalize_domain(url: str) -> str:
        # What: Handle empty URL safely.
        # Why: Prevent parser errors.
        if not url:
            return ""

        # What: Parse and normalize netloc.
        # Why: Convert domain to lowercase and trim whitespace.
        netloc = urlparse(url).netloc.lower().strip()
        # What: Drop common www prefix.
        # Why: Unify domain comparison against allowlist.
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # What: Return normalized domain.
        # Why: Reused by trust filter.
        return netloc

    # What: Check if URL belongs to trusted domains.
    # Why: Blocks untrusted/random sources.
    @staticmethod
    def _is_trusted_source(url: str) -> bool:
        # What: Normalize source domain.
        # Why: Stable comparison with allowlist.
        domain = WebAgent._normalize_domain(url)
        # What: Accept exact or subdomain match.
        # Why: Supports both root and nested trusted domains.
        return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in WebAgent.TRUSTED_DOMAINS)

    # What: Parse date string into date object.
    # Why: Enables freshness checks.
    @staticmethod
    def _parse_date(raw_date: str) -> date | None:
        # What: Handle missing date.
        # Why: Keep validator tolerant when provider omits date.
        if not raw_date:
            return None

        # What: Try supported date formats.
        # Why: Providers emit dates in different formats.
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                # What: Parse first 10 chars as date component.
                # Why: Ignores trailing time info if present.
                return datetime.strptime(raw_date[:10], fmt).date()
            # What: Ignore parse miss and try next format.
            # Why: Robust multi-format parsing.
            except ValueError:
                continue

        # What: Return None when parsing fails.
        # Why: Caller decides fallback behavior.
        return None

    # What: Determine if content is older than allowed threshold.
    # Why: Prevent stale citations in responses.
    @staticmethod
    def _is_too_old(raw_date: str) -> bool:
        # What: Parse provided date.
        # Why: Needed for age comparison.
        parsed = WebAgent._parse_date(raw_date)
        # What: Treat unknown date as not too old.
        # Why: Avoid over-filtering when date metadata is missing.
        if parsed is None:
            return False

        # What: Compute oldest allowed publication date.
        # Why: Enforce freshness window.
        oldest_allowed = date.today() - timedelta(days=WebAgent.MAX_RESULT_AGE_DAYS)
        # What: Return True if document predates allowed threshold.
        # Why: Filter stale sources.
        return parsed < oldest_allowed

    # What: Validate extracted results against trust/freshness/content rules.
    # Why: Keep only high-quality sources before summarization.
    @staticmethod
    def _validate_results(extracted_items: list[dict[str, str]]) -> list[dict[str, str]]:
        # What: Collected valid results.
        # Why: Output for summarizer.
        validated: list[dict[str, str]] = []
        # What: URL set for deduplication.
        # Why: Prevent repeated sources in final answer.
        seen_urls: set[str] = set()

        # What: Evaluate each normalized item.
        # Why: Apply all guardrail checks consistently.
        for item in extracted_items:
            url = item.get("url", "")
            content = item.get("content", "").strip()
            published_date = item.get("published_date", "")

            # What: Skip untrusted domains.
            # Why: Source quality control.
            if not WebAgent._is_trusted_source(url):
                continue

            # What: Skip duplicate URLs.
            # Why: Avoid duplicate evidence.
            if url in seen_urls:
                continue

            # What: Skip too-short content snippets.
            # Why: Weak content reduces summary quality.
            if len(content) < WebAgent.MIN_CONTENT_LENGTH:
                continue

            # What: Skip stale content.
            # Why: Keep responses timely.
            if WebAgent._is_too_old(published_date):
                continue

            # What: Record URL and keep item.
            # Why: Ensures unique validated source list.
            seen_urls.add(url)
            validated.append(item)

        # What: Return validated source items.
        # Why: Summarization should run on trusted evidence only.
        return validated

    # What: Summarize validated web items for the user query.
    # Why: Convert multiple sources into concise answer text.
    @staticmethod
    def _summarize(query: str, extracted_items: list[dict[str, str]]) -> str:
        # What: Handle no evidence case.
        # Why: Return explicit, safe empty-result response.
        if not extracted_items:
            return "No relevant web results found."

        # If OpenAI is unavailable, return a deterministic plain-text digest.
        # What: Use non-LLM fallback summary.
        # Why: Keep feature functional without OpenAI credentials.
        if settings.OPENAI_API_KEY is None:
            lines = ["Web summary:"]
            # What: Include up to 5 items in deterministic bullet form.
            # Why: Prevent very long fallback responses.
            for idx, item in enumerate(extracted_items[:5], start=1):
                lines.append(
                    f"{idx}. {item.get('title', 'Untitled')} - {item.get('content', '')} ({item.get('url', '')})"
                )
            return "\n".join(lines)

        # What: Build summarization prompt template.
        # Why: Instruct LLM to stay grounded on provided items.
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Summarize web search results accurately. Ground claims in the provided items and avoid speculation.",
                ),
                (
                    "human",
                    "User query:\n{query}\n\nExtracted web items:\n{items}\n\nWrite a concise, useful summary with key points.",
                ),
            ]
        )

        # What: Initialize LLM with deterministic settings.
        # Why: Reduce output variance for validation and reproducibility.
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        # What: Run prompt with compact JSON evidence payload.
        # Why: Provide structured context for accurate summarization.
        response = llm.invoke(prompt.format_messages(query=query, items=json.dumps(extracted_items[:8], indent=2)))

        # What: Return generated summary text.
        # Why: This becomes the web-agent answer.
        return response.content

    # What: Convenience API that returns answer string only.
    # Why: Keeps compatibility for callers expecting plain text.
    @staticmethod
    # What: Trace top-level web-agent invocations.
    # Why: Observability for production debugging.
    @opik.track(name="web_agent")
    def invoke(query: str, provider: str = "auto", k: int = 5) -> str:
        # What: Delegate to detailed method.
        # Why: Reuse common pipeline and metadata generation.
        result = WebAgent.invoke_with_details(query=query, provider=provider, k=k)
        # What: Return normalized string answer.
        # Why: Stable output contract.
        return str(result["answer"])

    # What: Full web-agent API with answer + evidence metadata.
    # Why: Supervisor/validator can inspect sources/provider details.
    @staticmethod
    # What: Separate trace name for detailed execution path.
    # Why: Distinguishes detailed call telemetry from simple invoke.
    @opik.track(name="web_agent_details")
    def invoke_with_details(query: str, provider: str = "auto", k: int = 5) -> dict[str, Any]:
        # What: Normalize provider selector.
        # Why: Accept case-insensitive provider argument.
        provider_lower = provider.lower()

        # What: Force Tavily route when requested.
        # Why: Deterministic provider control for testing/debugging.
        if provider_lower == "tavily":
            # What: Execute provider call behind circuit breaker.
            # Why: Protect from repeated provider failures.
            search_results = WebAgent._search_cb.call(WebAgent._search_tavily, query, k)
            selected_provider = "tavily"
        # What: Force Serper route when requested.
        # Why: Deterministic provider control for testing/debugging.
        elif provider_lower == "serper":
            search_results = WebAgent._search_cb.call(WebAgent._search_serper, query, k)
            selected_provider = "serper"
        else:
            # Auto mode: Tavily first, then Serper fallback.
            # What: Start with Tavily in auto mode.
            # Why: Preferred primary provider.
            selected_provider = "tavily"
            try:
                search_results = WebAgent._search_cb.call(WebAgent._search_tavily, query, k)
            # What: Swallow Tavily failure for fallback attempt.
            # Why: Auto mode should continue to secondary provider.
            except Exception:
                search_results = []

            # What: Fallback to Serper when Tavily yields nothing.
            # Why: Improve resilience and coverage.
            if not search_results:
                selected_provider = "serper"
                search_results = WebAgent._search_cb.call(WebAgent._search_serper, query, k)

        # What: Normalize raw provider results.
        # Why: Unify structure before validation.
        extracted_items = WebAgent._extract(search_results)
        # What: Apply trust/freshness/content guardrails.
        # Why: Keep only high-quality evidence.
        validated_items = WebAgent._validate_results(extracted_items)
        # What: Build final answer from validated items.
        # Why: Ensures summary is grounded in filtered sources.
        answer = WebAgent._summarize(query, validated_items)
        # What: Return answer plus diagnostics/evidence metadata.
        # Why: Supervisor/validator can use this for decisions and traceability.
        return {
            "answer": answer,
            "web_results": validated_items,
            "sources": [item.get("url", "") for item in validated_items if item.get("url")],
            "provider": selected_provider,
        }