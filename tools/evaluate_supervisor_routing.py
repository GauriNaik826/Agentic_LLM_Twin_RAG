# Keep type annotations deferred for cleaner typing/runtime compatibility.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Read/write CSV files.
import csv
# Work with filesystem-safe path objects.
from pathlib import Path
# Allow mixed types in metrics and payload dictionaries.
from typing import Any

# Import router under evaluation.
from llm_engineering.application.orchestration.router import QueryRouter


# Canonical supported route labels.
VALID_ROUTES = ("rag", "web", "twin_writer")


# Normalize route strings from dataset and model output into canonical labels.
def normalize_route(value: str) -> str:
    # Standardize spacing/case/hyphenation.
    normalized = value.strip().lower().replace(" ", "_")
    # Map common twin-writer variants to one canonical token.
    if normalized in {"twin", "twinwriter", "twin_writer", "twin-writer"}:
        return "twin_writer"
    # Keep already valid rag/web labels.
    if normalized in {"rag", "web"}:
        return normalized
    # Fallback to rag for unknown values.
    return "rag"


# Safe division helper to avoid ZeroDivisionError.
def safe_div(numerator: float, denominator: float) -> float:
    # Return 0.0 when denominator has no samples.
    if denominator == 0:
        return 0.0
    # Standard division path.
    return numerator / denominator


# Evaluate routing predictions against expected routes from CSV.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Instantiate the router once for all rows.
    router = QueryRouter()
    # Store row-level prediction details.
    rows: list[dict[str, str]] = []

    # Global counters for accuracy.
    total = 0
    correct = 0

    # confusion[expected][predicted] = count
    # Initialize full confusion matrix with zeros for all route pairs.
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in VALID_ROUTES}
        for expected in VALID_ROUTES
    }

    # Read dataset rows and run route prediction per query.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Parse query and normalize expected route.
            query = row["query"]
            expected = normalize_route(row["expected_route"])
            # Predict route with router and normalize output.
            predicted = normalize_route(router.classify(query))

            # Update global accuracy counts.
            total += 1
            if predicted == expected:
                correct += 1

            # Increment confusion matrix cell.
            confusion[expected][predicted] += 1

            # Save row-level output for audit/reporting.
            rows.append(
                {
                    "query": query,
                    "expected_route": expected,
                    "predicted_route": predicted,
                    "correct": str(predicted == expected),
                }
            )

    # Compute precision/recall/support per class from confusion matrix.
    metrics_by_class: dict[str, dict[str, float]] = {}
    for route in VALID_ROUTES:
        # True positives for this class.
        tp = confusion[route][route]
        # False positives: predicted as this class but expected other classes.
        fp = sum(confusion[other][route] for other in VALID_ROUTES if other != route)
        # False negatives: expected this class but predicted other classes.
        fn = sum(confusion[route][other] for other in VALID_ROUTES if other != route)

        # Standard precision/recall formulas with safe division.
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        # Store per-class metrics.
        metrics_by_class[route] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": sum(confusion[route].values()),
        }

    # Macro precision = arithmetic mean of class precisions.
    macro_precision = safe_div(
        sum(metrics_by_class[route]["precision"] for route in VALID_ROUTES),
        len(VALID_ROUTES),
    )
    # Macro recall = arithmetic mean of class recalls.
    macro_recall = safe_div(
        sum(metrics_by_class[route]["recall"] for route in VALID_ROUTES),
        len(VALID_ROUTES),
    )
    # Overall routing accuracy.
    accuracy = safe_div(correct, total)

    # Return row-level outputs, confusion matrix, and summary metrics.
    return {
        "rows": rows,
        "confusion_matrix": confusion,
        "metrics": {
            "routing_accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "per_class": metrics_by_class,
            "total": total,
            "correct": correct,
        },
    }


# Write row-level routing predictions to CSV.
def write_predictions(output_csv: Path, rows: list[dict[str, str]]) -> None:
    # Ensure target directory exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write output CSV with stable column order.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query", "expected_route", "predicted_route", "correct"],
        )
        # Emit header row.
        writer.writeheader()
        # Emit all row entries.
        writer.writerows(rows)


# Print confusion matrix in aligned text table format.
def print_confusion_matrix(confusion: dict[str, dict[str, int]]) -> None:
    # First row contains labels for predicted columns.
    header = ["expected\\predicted", *VALID_ROUTES]
    # Compute dynamic column widths for pretty alignment.
    widths = [max(len(header[0]), 18)]
    for route in VALID_ROUTES:
        col_max = max(len(route), *(len(str(confusion[row][route])) for row in VALID_ROUTES))
        widths.append(max(col_max, 12))

    # Local helper that pads and joins cells for one table row.
    def format_row(values: list[str]) -> str:
        padded = [values[i].ljust(widths[i]) for i in range(len(values))]
        return " | ".join(padded)

    # Print table header and separator.
    print("Confusion Matrix")
    print(format_row(header))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    # Print one row per expected class.
    for expected in VALID_ROUTES:
        values = [expected, *(str(confusion[expected][predicted]) for predicted in VALID_ROUTES)]
        print(format_row(values))


# CLI entrypoint for running routing evaluation from terminal.
def main() -> None:
    # Build argument parser.
    parser = argparse.ArgumentParser(description="Evaluate supervisor routing predictions against a labeled CSV.")
    # Input dataset option.
    parser.add_argument(
        "--csv",
        default="data/artifacts/routing_dataset.csv",
        help="Input CSV path with query and expected_route columns.",
    )
    # Output predictions file option.
    parser.add_argument(
        "--out",
        default="data/artifacts/routing_predictions.csv",
        help="Output CSV path for row-level routing predictions.",
    )
    # Parse CLI arguments.
    args = parser.parse_args()

    # Run evaluation and write row-level predictions.
    result = evaluate(Path(args.csv))
    write_predictions(Path(args.out), result["rows"])

    # Print high-level summary.
    metrics = result["metrics"]
    print("Supervisor Routing Evaluation")
    print(f"Dataset: {args.csv}")
    print(f"Queries: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print("")
    print("Metrics")
    print(f"- Routing Accuracy: {metrics['routing_accuracy']:.2%}")
    print(f"- Precision (Macro): {metrics['macro_precision']:.2%}")
    print(f"- Recall (Macro): {metrics['macro_recall']:.2%}")
    print("")
    print("Per-class")
    for route in VALID_ROUTES:
        route_metrics = metrics["per_class"][route]
        print(
            f"- {route}: precision={route_metrics['precision']:.2%}, "
            f"recall={route_metrics['recall']:.2%}, support={route_metrics['support']}"
        )
    print("")

    # Print formatted confusion matrix.
    print_confusion_matrix(result["confusion_matrix"])
    print("")
    # Print output file path.
    print(f"Detailed predictions written to: {args.out}")


# Execute main only when run directly as a script.
if __name__ == "__main__":
    main()
