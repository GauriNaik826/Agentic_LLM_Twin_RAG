# Keep annotations deferred for smoother typing/runtime compatibility.
from __future__ import annotations

# Parse CLI arguments.
import argparse
# Read and write CSV files.
import csv
# Regex utilities for citation URL detection.
import re
# Compute mean latency across rows.
import statistics
# High-resolution timing for search latency measurement.
import time
# Path-safe filesystem operations.
from pathlib import Path
# Allow mixed-type dictionaries for rows/metrics.
from typing import Any

# Web agent under evaluation.
from llm_engineering.application.agents.web_agent import WebAgent


# URL regex used to detect citation links in answer text.
URL_PATTERN = re.compile(r"https?://\\S+", re.IGNORECASE)


# Safe division helper to prevent ZeroDivisionError.
def safe_div(numerator: float, denominator: float) -> float:
    # Return 0 when denominator has no samples.
    if denominator == 0:
        return 0.0
    # Normal division path.
    return numerator / denominator


# Evaluate web agent behavior for citation/trust/latency/coverage metrics.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Store row-level output details for CSV export.
    rows: list[dict[str, Any]] = []

    # Query-level volume counters.
    total_queries = 0
    answered_queries = 0
    answers_with_citations = 0

    # Source-level trust counters.
    total_sources = 0
    trusted_sources = 0

    # Per-query latency samples in milliseconds.
    latencies_ms: list[float] = []

    # Open dataset CSV and process each query row.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Read user query and increment total query count.
            query = row["query"]
            total_queries += 1

            # Measure web-agent end-to-end response latency.
            start = time.perf_counter()
            result = WebAgent.invoke_with_details(query)
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(latency_ms)

            # Parse key fields from web-agent output.
            answer = str(result.get("answer", "")).strip()
            sources = [src for src in result.get("sources", []) if isinstance(src, str) and src.strip()]
            provider = str(result.get("provider", "unknown"))

            # Decide if we consider this query answered.
            has_answer = bool(answer) and answer != "No relevant web results found."
            # Detect citation via source list or URL embedded in answer text.
            has_citation = len(sources) > 0 or bool(URL_PATTERN.search(answer))

            # Update answered/citation counters at query level.
            if has_answer:
                answered_queries += 1
                if has_citation:
                    answers_with_citations += 1

            # Update source trust counters.
            for src in sources:
                total_sources += 1
                if WebAgent._is_trusted_source(src):
                    trusted_sources += 1

            # Save row-level diagnostics.
            rows.append(
                {
                    "query": query,
                    "provider": provider,
                    "answer": answer,
                    "sources_count": len(sources),
                    "has_answer": has_answer,
                    "has_citation": has_citation,
                    "search_latency_ms": round(latency_ms, 2),
                }
            )

    # Compute aggregate evaluation metrics.
    coverage = safe_div(answered_queries, total_queries)
    citation_accuracy = safe_div(answers_with_citations, answered_queries)
    trusted_source_percent = safe_div(trusted_sources, total_sources)
    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0

    # Build rounded metric payload.
    metrics = {
        "citation_accuracy": round(citation_accuracy, 4),
        "trusted_sources_percent": round(trusted_source_percent, 4),
        "search_latency_ms_avg": round(avg_latency, 2),
        "coverage": round(coverage, 4),
        "total_queries": total_queries,
        "answered_queries": answered_queries,
        "total_sources": total_sources,
    }

    # Return row-level details and summary metrics.
    return {"rows": rows, "metrics": metrics}


# Write detailed row-level web-agent results to CSV.
def write_results(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Ensure destination folder exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Stable output column order.
    fieldnames = [
        "query",
        "provider",
        "answer",
        "sources_count",
        "has_answer",
        "has_citation",
        "search_latency_ms",
    ]
    # Write CSV file with header and all rows.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# CLI entrypoint for running web-agent evaluation from terminal.
def main() -> None:
    # Build argument parser.
    parser = argparse.ArgumentParser(description="Evaluate Web Agent metrics for citations, trust, latency, and coverage.")
    # Input dataset path argument.
    parser.add_argument(
        "--csv",
        default="data/artifacts/web_agent_eval_dataset.csv",
        help="CSV with query column.",
    )
    # Output predictions path argument.
    parser.add_argument(
        "--out",
        default="data/artifacts/web_agent_eval_results.csv",
        help="Output CSV path for row-level web agent outputs.",
    )
    # Parse CLI arguments.
    args = parser.parse_args()

    # Run evaluation and persist detailed results.
    result = evaluate(Path(args.csv))
    write_results(Path(args.out), result["rows"])

    # Print concise terminal summary.
    metrics = result["metrics"]
    print("Web Agent Evaluation")
    print(f"Dataset: {args.csv}")
    print("")
    print("Metrics")
    print(f"- Citation Accuracy: {metrics['citation_accuracy']:.2%}")
    print(f"- Trusted Sources %: {metrics['trusted_sources_percent']:.2%}")
    print(f"- Search Latency (avg ms): {metrics['search_latency_ms_avg']}")
    print(f"- Coverage: {metrics['coverage']:.2%}")
    print("")
    print("Volume")
    print(f"- Total Queries: {metrics['total_queries']}")
    print(f"- Answered Queries: {metrics['answered_queries']}")
    print(f"- Total Sources: {metrics['total_sources']}")
    print("")
    print(f"Detailed row-level output: {args.out}")


# Execute main only when this file is run directly.
if __name__ == "__main__":
    main()
