# Keep type annotations deferred to runtime for smoother typing compatibility.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Read and write CSV files.
import csv
# Measure elapsed execution time for recovery scenarios.
import time
# Work with filesystem paths in a cross-platform way.
from pathlib import Path
# Allow mixed value types for row and metric payloads.
from typing import Any
# Monkeypatch functions/methods to force specific failure scenarios.
from unittest.mock import patch

# Import Twin Writer class for direct method patching.
from llm_engineering.application.agents.twin_writer import TwinWriterAgent
# Import Web agent timeout exception for forced timeout simulation.
from llm_engineering.application.agents.web_agent import WebAgent, WebSearchTimeoutError
# Import ValidationResult structure to mock validator output consistently.
from llm_engineering.application.guardrails.output_validator import ValidationResult
# Import shared state singleton to reset and inspect post-run metadata.
from llm_engineering.application.orchestration.state import shared_supervisor_state
# Import supervisor orchestrator under test.
from llm_engineering.application.orchestration.supervisor import Supervisor


# Scenario ID used to test: retriever returns no docs, so RAG should fallback to web.
SCENARIO_RETRIEVER_NO_DOCS = "retriever_no_documents"
# Scenario ID used to test: web search timeout, so web route should fallback to twin writer.
SCENARIO_WEB_TIMEOUT = "web_timeout"
# Scenario ID used to test: twin writer failure, so safe response should be returned.
SCENARIO_TWIN_WRITER_FAILURE = "twin_writer_failure"

# Expected safe fallback message from the twin writer node when invoke raises an exception.
SAFE_TWIN_FAILURE_RESPONSE = "I cannot generate that response right now. Please try a different request."


# Safe division helper to prevent division-by-zero in aggregate metrics.
def safe_div(numerator: float, denominator: float) -> float:
    # If denominator has no samples, define metric as 0.
    if denominator == 0:
        return 0.0
    # Standard division path.
    return numerator / denominator


# Load recovery scenarios from CSV into normalized dictionaries.
def parse_scenarios(csv_path: Path) -> list[dict[str, str]]:
    # Destination list for all parsed scenario rows.
    scenarios: list[dict[str, str]] = []
    # Open input file and iterate rows by header names.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Normalize spacing/case for deterministic comparisons.
            scenarios.append(
                {
                    "scenario": row["scenario"].strip(),
                    "query": row["query"].strip(),
                    "forced_route": row["forced_route"].strip().lower(),
                    "expected_fallback": row["expected_fallback"].strip().lower(),
                }
            )
    # Return list of normalized scenarios.
    return scenarios


# Return a passing validator result so validator logic does not interfere with recovery-path testing.
def _baseline_validation_result() -> ValidationResult:
    # Construct result object where all checks pass.
    return ValidationResult(
        passed=True,
        confidence=0.99,
        reason=None,
        checks={
            "empty_answer": False,
            "grounding_ok": True,
            "citation_ok": True,
            "style_ok": True,
            "toxicity_ok": True,
            "length_ok": True,
            "hallucination_ok": True,
        },
    )


# Execute exactly one forced recovery scenario and capture outcome details.
def run_scenario(supervisor: Supervisor, scenario: dict[str, str]) -> dict[str, Any]:
    # Pull normalized scenario inputs.
    scenario_name = scenario["scenario"]
    query = scenario["query"]
    forced_route = scenario["forced_route"]
    expected_fallback = scenario["expected_fallback"]

    # Start from a clean shared state so previous runs do not leak metadata.
    shared_supervisor_state.reset()

    # Patch router to force target route and patch validator to always pass.
    with patch(
        "llm_engineering.application.orchestration.router.QueryRouter.classify",
        return_value=forced_route,
    ), patch(
        "llm_engineering.application.guardrails.output_validator.OutputValidator.validate",
        return_value=_baseline_validation_result(),
    ):
        # Start timing immediately before scenario-specific invoke.
        started = time.perf_counter()

        # Scenario 1: force retriever empty -> RAG should use Web fallback.
        if scenario_name == SCENARIO_RETRIEVER_NO_DOCS:
            with patch(
                "llm_engineering.application.rag.retriever.ContextRetriever.search",
                return_value=[],
            ), patch(
                "llm_engineering.application.agents.web_agent.WebAgent.invoke",
                return_value="WEB_FALLBACK_ANSWER",
            ):
                # Trigger supervisor flow.
                answer = supervisor.invoke(query)

        # Scenario 2: force web timeout -> supervisor should fallback to Twin Writer.
        elif scenario_name == SCENARIO_WEB_TIMEOUT:
            with patch(
                "llm_engineering.application.agents.web_agent.WebAgent.invoke_with_details",
                side_effect=WebSearchTimeoutError("forced timeout"),
            ), patch.object(
                TwinWriterAgent,
                "invoke",
                return_value="TWIN_WRITER_FALLBACK_ANSWER",
            ):
                # Trigger supervisor flow.
                answer = supervisor.invoke(query)

        # Scenario 3: force twin writer failure -> twin node should return safe response.
        elif scenario_name == SCENARIO_TWIN_WRITER_FAILURE:
            with patch.object(
                TwinWriterAgent,
                "invoke",
                side_effect=RuntimeError("forced twin writer failure"),
            ):
                # Trigger supervisor flow.
                answer = supervisor.invoke(query)

        # Fail fast on unknown scenario IDs.
        else:
            raise ValueError(f"Unsupported scenario: {scenario_name}")

        # Compute elapsed recovery execution time in milliseconds.
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Read final metadata written by supervisor.
    final_state = shared_supervisor_state.get()
    metadata = dict(final_state.metadata)

    # Initialize fallback detection outputs.
    fallback_detected = False
    fallback_target = "none"

    # Decide whether expected fallback happened based on metadata and answer.
    if scenario_name == SCENARIO_RETRIEVER_NO_DOCS:
        fallback_detected = bool(metadata.get("rag_used_web_fallback"))
        fallback_target = "web" if fallback_detected else "none"

    elif scenario_name == SCENARIO_WEB_TIMEOUT:
        fallback_detected = metadata.get("fallback_from") == "web_timeout" and metadata.get("executed_agent") == "twin_writer"
        fallback_target = "twin_writer" if fallback_detected else "none"

    elif scenario_name == SCENARIO_TWIN_WRITER_FAILURE:
        fallback_detected = answer == SAFE_TWIN_FAILURE_RESPONSE
        fallback_target = "safe_response" if fallback_detected else "none"

    # Scenario is successful when actual fallback equals expected fallback label.
    success = fallback_target == expected_fallback

    # Return detailed row output for this scenario.
    return {
        "scenario": scenario_name,
        "query": query,
        "forced_route": forced_route,
        "expected_fallback": expected_fallback,
        "actual_fallback": fallback_target,
        "success": success,
        "fallback_detected": fallback_detected,
        "recovery_time_ms": round(elapsed_ms, 2),
        "answer": answer,
    }


# Run all scenarios and compute aggregate recovery metrics.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Build supervisor once and reuse across scenarios.
    supervisor = Supervisor()
    # Load all scenario rows.
    scenarios = parse_scenarios(csv_path)

    # Row-level and aggregate containers.
    rows: list[dict[str, Any]] = []
    recovery_times: list[float] = []
    success_count = 0
    fallback_count = 0

    # Execute each scenario and accumulate counters.
    for scenario in scenarios:
        result = run_scenario(supervisor, scenario)
        rows.append(result)
        recovery_times.append(float(result["recovery_time_ms"]))

        if result["success"]:
            success_count += 1
        if result["fallback_detected"]:
            fallback_count += 1

    # Number of executed scenarios.
    total = len(rows)

    # Compute aggregate Step 5 metrics.
    metrics = {
        "recovery_success_rate": round(safe_div(success_count, total), 4),
        "average_recovery_time_ms": round(safe_div(sum(recovery_times), len(recovery_times)), 2),
        "fallback_frequency": round(safe_div(fallback_count, total), 4),
        "total_scenarios": total,
    }

    # Return both row-level outputs and summary metrics.
    return {"rows": rows, "metrics": metrics}


# Write scenario-level recovery results to CSV.
def write_results(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Ensure destination directory exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Keep stable output column order.
    fieldnames = [
        "scenario",
        "query",
        "forced_route",
        "expected_fallback",
        "actual_fallback",
        "success",
        "fallback_detected",
        "recovery_time_ms",
        "answer",
    ]
    # Write header and row entries.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# CLI entrypoint for running recovery policy evaluation from terminal.
def main() -> None:
    # Build parser and expose dataset/output arguments.
    parser = argparse.ArgumentParser(description="Evaluate Supervisor recovery policies across forced failure scenarios.")
    parser.add_argument(
        "--csv",
        default="data/artifacts/recovery_policy_scenarios.csv",
        help="Input CSV with recovery scenarios.",
    )
    parser.add_argument(
        "--out",
        default="data/artifacts/recovery_policy_eval_results.csv",
        help="Output CSV path for row-level recovery results.",
    )
    # Parse CLI args.
    args = parser.parse_args()

    # Run evaluation and persist detailed scenario outputs.
    result = evaluate(Path(args.csv))
    write_results(Path(args.out), result["rows"])

    # Print compact metric summary to console.
    metrics = result["metrics"]
    print("Recovery Policy Evaluation")
    print(f"Dataset: {args.csv}")
    print("")
    print("Metrics")
    print(f"- Recovery Success Rate: {metrics['recovery_success_rate']:.2%}")
    print(f"- Average Recovery Time (ms): {metrics['average_recovery_time_ms']}")
    print(f"- Fallback Frequency: {metrics['fallback_frequency']:.2%}")
    print("")
    print(f"Total Scenarios: {metrics['total_scenarios']}")
    print(f"Detailed row-level output: {args.out}")


# Execute main only when invoked as a script.
if __name__ == "__main__":
    main()
