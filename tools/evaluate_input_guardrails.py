# Keep type annotations as deferred strings for cleaner forward typing behavior.
from __future__ import annotations

# Parse CLI arguments.
import argparse
# Read and write CSV files.
import csv
# Work with filesystem paths safely.
from pathlib import Path
# Allow mixed value types in dict payloads.
from typing import Any

# Import the input guardrail and the two block exceptions it can raise.
from llm_engineering.application.guardrails.input_guardrail import (
    InputGuardrail,
    UnsafePromptException,
    UnsupportedRequestException,
)


# Convert common truthy strings into a boolean.
def parse_bool(value: str) -> bool:
    # Normalize spacing and case before checking truthy tokens.
    return value.strip().lower() in {"1", "true", "yes", "y"}


# Divide safely to avoid ZeroDivisionError when denominator is zero.
def safe_div(numerator: float, denominator: float) -> float:
    # Return 0.0 when the metric denominator has no samples.
    if denominator == 0:
        return 0.0
    # Standard division when denominator is valid.
    return numerator / denominator


# Run guardrail evaluation over a labeled CSV dataset.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Create a guardrail instance for all rows.
    guardrail = InputGuardrail()

    # Collect per-row prediction outputs for detailed reporting.
    rows: list[dict[str, Any]] = []

    # PII confusion counters.
    pii_tp = 0
    pii_fp = 0
    pii_fn = 0

    # Prompt-injection counters.
    inj_total = 0
    inj_correct = 0
    inj_fp = 0
    inj_fn = 0

    # Open dataset CSV in read mode.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        # Read rows by header names.
        reader = csv.DictReader(handle)
        # Evaluate one query per row.
        for row in reader:
            # Input query and expected labels from dataset.
            query = row["query"]
            expected_action = row["expected_action"].strip().upper()
            expected_pii = parse_bool(row["expected_pii"])
            expected_injection = parse_bool(row["expected_injection"])

            # Initialize predicted outputs.
            predicted_action = "PASS"
            predicted_pii = False
            predicted_injection = False
            blocked_reason = ""
            redacted_query = ""

            # Run guardrail processing and map outputs to prediction fields.
            try:
                redacted_query, metadata = guardrail.process(query)
                predicted_pii = bool(metadata.get("pii_detected", False))
                # If PII is detected and masked, action becomes PII_MASK.
                if predicted_pii:
                    predicted_action = "PII_MASK"
            # Injection block maps to BLOCK + injection detected.
            except UnsafePromptException:
                predicted_action = "BLOCK"
                predicted_injection = True
                blocked_reason = "prompt_injection"
            # Unsupported request also maps to BLOCK but not prompt injection.
            except UnsupportedRequestException:
                predicted_action = "BLOCK"
                blocked_reason = "unsupported_request"

            # Update PII confusion counts.
            if expected_pii and predicted_pii:
                pii_tp += 1
            elif (not expected_pii) and predicted_pii:
                pii_fp += 1
            elif expected_pii and (not predicted_pii):
                pii_fn += 1

            # Update injection classification counters.
            inj_total += 1
            if predicted_injection == expected_injection:
                inj_correct += 1
            if predicted_injection and (not expected_injection):
                inj_fp += 1
            if expected_injection and (not predicted_injection):
                inj_fn += 1

            # Store row-level expected vs predicted output for auditability.
            rows.append(
                {
                    "query": query,
                    "expected_action": expected_action,
                    "predicted_action": predicted_action,
                    "expected_pii": expected_pii,
                    "predicted_pii": predicted_pii,
                    "expected_injection": expected_injection,
                    "predicted_injection": predicted_injection,
                    "blocked_reason": blocked_reason,
                    "redacted_query": redacted_query,
                }
            )

    # Compute final aggregated metrics.
    pii_precision = safe_div(pii_tp, pii_tp + pii_fp)
    pii_recall = safe_div(pii_tp, pii_tp + pii_fn)
    injection_accuracy = safe_div(inj_correct, inj_total)

    # Return both row-level results and summary metrics.
    return {
        "rows": rows,
        "metrics": {
            "pii_detection_precision": round(pii_precision, 4),
            "pii_detection_recall": round(pii_recall, 4),
            "prompt_injection_detection_accuracy": round(injection_accuracy, 4),
            "false_positives": inj_fp,
            "false_negatives": inj_fn,
        },
    }


# Write detailed predictions to an output CSV.
def write_results(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Define stable output column order.
    fieldnames = [
        "query",
        "expected_action",
        "predicted_action",
        "expected_pii",
        "predicted_pii",
        "expected_injection",
        "predicted_injection",
        "blocked_reason",
        "redacted_query",
    ]
    # Ensure parent folder exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write file in UTF-8 with standard CSV handling.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        # Create DictWriter bound to declared columns.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Write CSV header row.
        writer.writeheader()
        # Write all data rows.
        writer.writerows(rows)


# CLI entrypoint for running evaluation from terminal.
def main() -> None:
    # Build argument parser and help text.
    parser = argparse.ArgumentParser(description="Evaluate input guardrail behavior against a labeled CSV dataset.")
    # Input dataset path argument.
    parser.add_argument(
        "--csv",
        default="data/artifacts/input_guardrail_eval.csv",
        help="Input CSV path containing query and expected labels.",
    )
    # Output results path argument.
    parser.add_argument(
        "--out",
        default="data/artifacts/input_guardrail_eval_results.csv",
        help="Output CSV path for row-level predictions.",
    )
    # Parse CLI values.
    args = parser.parse_args()

    # Run evaluation.
    result = evaluate(Path(args.csv))
    # Persist row-level results.
    write_results(Path(args.out), result["rows"])

    # Print compact console summary.
    print("Input Guardrail Evaluation")
    print(f"Dataset: {args.csv}")
    print(f"Rows: {len(result['rows'])}")
    print("")
    print("Metrics")
    print(f"- PII Detection Precision: {result['metrics']['pii_detection_precision']:.2%}")
    print(f"- PII Detection Recall: {result['metrics']['pii_detection_recall']:.2%}")
    print(
        "- Prompt Injection Detection Accuracy: "
        f"{result['metrics']['prompt_injection_detection_accuracy']:.2%}"
    )
    print(f"- False Positives: {result['metrics']['false_positives']}")
    print(f"- False Negatives: {result['metrics']['false_negatives']}")
    print("")
    print(f"Detailed row-level output: {args.out}")


# Run main only when executed as a script.
if __name__ == "__main__":
    main()
