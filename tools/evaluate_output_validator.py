# Keep type annotations deferred to improve compatibility with modern typing usage.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Read and write CSV data.
import csv
# Parse JSON strings from CSV columns.
import json
# Work with filesystem paths safely.
from pathlib import Path
# Allow mixed types in metrics and row payloads.
from typing import Any

# Import the validator under test.
from llm_engineering.application.guardrails.output_validator import OutputValidator


# Allowed label values for expected final decision.
VERDICTS = {"ACCEPT", "REJECT"}


# Convert common truthy string values into boolean.
def parse_bool(value: str) -> bool:
    # Normalize case and whitespace before comparison.
    return value.strip().lower() in {"1", "true", "yes", "y"}


# Divide safely to avoid division by zero.
def safe_div(numerator: float, denominator: float) -> float:
    # Return 0 when there are no samples for a metric denominator.
    if denominator == 0:
        return 0.0
    # Normal division path.
    return numerator / denominator


# Parse web_results_json column into a list of dictionaries.
def parse_web_results(raw: str) -> list[dict[str, Any]]:
    # Remove surrounding whitespace from input text.
    text = raw.strip()
    # Empty input means no web results.
    if not text:
        return []
    # Try decoding JSON string.
    try:
        data = json.loads(text)
    # Invalid JSON should be treated as empty results rather than crashing.
    except json.JSONDecodeError:
        return []
    # Accept only list-shaped JSON payloads.
    if isinstance(data, list):
        # Keep only dictionary entries from the list.
        cleaned: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                cleaned.append(item)
        return cleaned
    # Any non-list JSON payload is considered invalid for this field.
    return []


# Run validator predictions and compare them to expected labels.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Create validator instance once for the full dataset.
    validator = OutputValidator()

    # Store row-level prediction details for output CSV.
    rows: list[dict[str, Any]] = []

    # Overall verdict counters.
    total = 0
    correct = 0

    # Hallucination-specific counters.
    halluc_total = 0
    halluc_detected = 0

    # Citation detection counters.
    citation_total = 0
    citation_correct = 0

    # Style-validation counters (twin_writer rows only).
    style_total = 0
    style_correct = 0

    # Open input CSV in UTF-8 and parse by header names.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # Evaluate one labeled example per row.
        for row in reader:
            # Read input fields used by the validator.
            route = row["route"].strip().lower()
            query = row["query"]
            answer = row["answer"]
            retrieved_context = row["retrieved_context"]
            web_results = parse_web_results(row.get("web_results_json", ""))

            # Read expected labels from dataset.
            expected_verdict = row["expected_verdict"].strip().upper()
            # Validate expected verdict label to catch dataset mistakes early.
            if expected_verdict not in VERDICTS:
                raise ValueError(f"Invalid expected_verdict: {expected_verdict}")

            expected_hallucination = parse_bool(row["expected_hallucination"])
            expected_citation = parse_bool(row["expected_citation"])
            expected_style = parse_bool(row["expected_style"])

            # Run actual output validator for this row.
            result = validator.validate(
                route=route,
                query=query,
                answer=answer,
                retrieved_context=retrieved_context,
                web_results=web_results,
            )

            # Convert validator output into comparable prediction labels.
            predicted_verdict = "ACCEPT" if result.passed else "REJECT"
            predicted_hallucination = not bool(result.checks.get("hallucination_ok", True))
            predicted_citation = bool(result.checks.get("citation_ok", True))
            predicted_style = bool(result.checks.get("style_ok", True))

            # Update overall accuracy counters.
            total += 1
            if predicted_verdict == expected_verdict:
                correct += 1

            # Update hallucination detection counters.
            halluc_total += 1 if expected_hallucination else 0
            if expected_hallucination and predicted_hallucination:
                halluc_detected += 1

            # Update citation detection counters.
            citation_total += 1
            if predicted_citation == expected_citation:
                citation_correct += 1

            # Update style validation counters only for twin_writer route.
            style_total += 1 if route == "twin_writer" else 0
            if route == "twin_writer" and predicted_style == expected_style:
                style_correct += 1

            # Append row-level prediction details for downstream analysis.
            rows.append(
                {
                    "route": route,
                    "query": query,
                    "expected_verdict": expected_verdict,
                    "predicted_verdict": predicted_verdict,
                    "expected_hallucination": expected_hallucination,
                    "predicted_hallucination": predicted_hallucination,
                    "expected_citation": expected_citation,
                    "predicted_citation": predicted_citation,
                    "expected_style": expected_style,
                    "predicted_style": predicted_style,
                    "confidence": result.confidence,
                    "reason": result.reason or "",
                }
            )

    # Compute aggregate metric values.
    metrics = {
        "validator_accuracy": round(safe_div(correct, total), 4),
        "hallucination_detection_rate": round(safe_div(halluc_detected, halluc_total), 4),
        "citation_detection": round(safe_div(citation_correct, citation_total), 4),
        "style_validation_accuracy": round(safe_div(style_correct, style_total), 4),
        "total_rows": total,
    }

    # Return full detailed rows plus summary metrics.
    return {"rows": rows, "metrics": metrics}


# Write row-level evaluation outputs into CSV.
def write_results(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Ensure destination folder exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Stable output column order.
    fieldnames = [
        "route",
        "query",
        "expected_verdict",
        "predicted_verdict",
        "expected_hallucination",
        "predicted_hallucination",
        "expected_citation",
        "predicted_citation",
        "expected_style",
        "predicted_style",
        "confidence",
        "reason",
    ]
    # Open output file and write CSV contents.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Write header row.
        writer.writeheader()
        # Write all prediction rows.
        writer.writerows(rows)


# Script entrypoint for running evaluation from terminal.
def main() -> None:
    # Build CLI parser.
    parser = argparse.ArgumentParser(description="Evaluate OutputValidator against labeled answer examples.")
    # Input dataset argument.
    parser.add_argument(
        "--csv",
        default="data/artifacts/output_validator_eval_dataset.csv",
        help="Input CSV for validator evaluation.",
    )
    # Output predictions argument.
    parser.add_argument(
        "--out",
        default="data/artifacts/output_validator_eval_results.csv",
        help="Output CSV path for row-level validator predictions.",
    )
    # Parse CLI values.
    args = parser.parse_args()

    # Execute evaluation pipeline.
    result = evaluate(Path(args.csv))
    # Persist row-level details.
    write_results(Path(args.out), result["rows"])

    # Print concise metrics summary to console.
    metrics = result["metrics"]
    print("Output Validator Evaluation")
    print(f"Dataset: {args.csv}")
    print("")
    print("Metrics")
    print(f"- Validator Accuracy: {metrics['validator_accuracy']:.2%}")
    print(f"- Hallucination Detection Rate: {metrics['hallucination_detection_rate']:.2%}")
    print(f"- Citation Detection: {metrics['citation_detection']:.2%}")
    print(f"- Style Validation Accuracy: {metrics['style_validation_accuracy']:.2%}")
    print("")
    print(f"Total Rows: {metrics['total_rows']}")
    print(f"Detailed row-level output: {args.out}")


# Execute main only when file is run as a script.
if __name__ == "__main__":
    main()
