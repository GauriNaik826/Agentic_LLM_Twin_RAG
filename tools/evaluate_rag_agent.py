# Keep annotations deferred, which helps with typing compatibility in modern Python.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Read/write CSV files.
import csv
# Compute means for aggregate metrics.
import statistics
# Measure latency with high-resolution timers.
import time
# Work with filesystem paths safely.
from pathlib import Path
# Support mixed-type dictionaries for result payloads.
from typing import Any

# Import RAG agent utilities used for dedup and generation calls.
from llm_engineering.application.agents.rag_agent import RAGAgent
# Import retriever to collect context docs for latency/similarity analysis.
from llm_engineering.application.rag.retriever import ContextRetriever


# Divide safely to avoid zero-division in metrics.
def safe_div(numerator: float, denominator: float) -> float:
    # Return 0.0 when denominator has no samples.
    if denominator == 0:
        return 0.0
    # Normal division path.
    return numerator / denominator


# Compute average similarity score from retrieved document metadata.
def average_similarity_from_docs(docs: list[Any]) -> float:
    # Collect valid numeric scores.
    scores: list[float] = []
    # Iterate through each retrieved document.
    for doc in docs:
        # Read metadata safely even if field is missing/None.
        metadata = getattr(doc, "metadata", {}) or {}
        # Accept either "score" or "similarity" key conventions.
        score = metadata.get("score", metadata.get("similarity"))
        # Keep only numeric values.
        if isinstance(score, (int, float)):
            scores.append(float(score))
    # No valid scores means similarity is unknown, treated as 0.
    if not scores:
        return 0.0
    # Return arithmetic mean similarity.
    return float(sum(scores) / len(scores))


# Run RAGAS metrics over question/answer/context/ground-truth arrays.
def run_ragas(questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]) -> dict[str, float]:
    # Import RAGAS lazily so script can load even when package is missing.
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall, faithfulness

        # RAGAS has used two possible names across versions for answer relevance.
        try:
            from ragas.metrics import answer_relevancy as answer_relevance_metric
        except Exception:
            from ragas.metrics import answer_relevance as answer_relevance_metric
    # Convert import/setup issues into a clear actionable error.
    except Exception as exc:
        raise RuntimeError(
            "RAGAS is required for this evaluation. Install dependencies and rerun."
        ) from exc

    # Build Hugging Face Dataset expected by RAGAS.
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Execute selected RAGAS metrics.
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevance_metric],
    )

    # Extract only the expected metric keys if present.
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_relevance"]
    scores: dict[str, float] = {}
    for name in metric_names:
        if name in result:
            scores[name] = float(result[name])

    # Return normalized score dictionary.
    return scores


# Evaluate RAG agent quality and operations against a labeled CSV dataset.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Use real retriever for latency and retrieval diagnostics.
    retriever = ContextRetriever(mock=False)

    # Store row-level outputs for CSV export.
    rows: list[dict[str, Any]] = []
    # Store arrays required by RAGAS evaluation.
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    # Store operational metric components.
    retrieval_latencies_ms: list[float] = []
    avg_similarities: list[float] = []
    duplicate_removed_counts: list[int] = []

    # Open and iterate input dataset rows.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Read query and reference answer.
            query = row["query"]
            ground_truth = row["ground_truth"]

            # Measure retrieval time only for retriever step.
            start = time.perf_counter()
            retrieved_docs = retriever.search(query, k=3)
            retrieval_latency_ms = (time.perf_counter() - start) * 1000.0

            # Deduplicate retrieval results for cleaner context.
            dedup_docs = RAGAgent._deduplicate_docs(retrieved_docs)
            # Count how many duplicates were removed.
            duplicate_removed = max(0, len(retrieved_docs) - len(dedup_docs))
            # Compute average similarity from deduped docs.
            avg_similarity = average_similarity_from_docs(dedup_docs)

            # Run RAG answer generation without web fallback to isolate RAG behavior.
            rag_result = RAGAgent.invoke_with_details(query, allow_web_fallback=False)
            answer = str(rag_result.get("answer", ""))

            # Extract plain context strings from deduplicated docs for RAGAS.
            row_contexts = [getattr(doc, "content", "") for doc in dedup_docs if getattr(doc, "content", "")]

            # Save row-level diagnostic output.
            rows.append(
                {
                    "query": query,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                    "average_similarity": round(avg_similarity, 4),
                    "raw_docs": len(retrieved_docs),
                    "deduplicated_docs": len(dedup_docs),
                    "duplicate_removed": duplicate_removed,
                    "used_web_fallback": bool(rag_result.get("used_web_fallback", False)),
                }
            )

            # Append row values to arrays used by RAGAS and aggregates.
            questions.append(query)
            answers.append(answer)
            contexts.append(row_contexts)
            ground_truths.append(ground_truth)
            retrieval_latencies_ms.append(retrieval_latency_ms)
            avg_similarities.append(avg_similarity)
            duplicate_removed_counts.append(duplicate_removed)

    # Run semantic quality evaluation with RAGAS.
    ragas_scores = run_ragas(questions, answers, contexts, ground_truths)

    # Compute aggregate operational metrics.
    aggregate = {
        "retrieval_latency_ms_avg": round(statistics.mean(retrieval_latencies_ms), 2) if retrieval_latencies_ms else 0.0,
        "average_similarity": round(statistics.mean(avg_similarities), 4) if avg_similarities else 0.0,
        "duplicate_removal_avg": round(statistics.mean(duplicate_removed_counts), 2)
        if duplicate_removed_counts
        else 0.0,
    }

    # Return row-level outputs, RAGAS scores, and aggregate operational metrics.
    return {
        "rows": rows,
        "ragas": ragas_scores,
        "aggregate": aggregate,
    }


# Write row-level predictions to CSV.
def write_predictions(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Ensure destination directory exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Define stable output column order.
    fieldnames = [
        "query",
        "ground_truth",
        "answer",
        "retrieval_latency_ms",
        "average_similarity",
        "raw_docs",
        "deduplicated_docs",
        "duplicate_removed",
        "used_web_fallback",
    ]
    # Open output file and write rows.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Write header row.
        writer.writeheader()
        # Write all data rows.
        writer.writerows(rows)


# CLI entrypoint for terminal usage.
def main() -> None:
    # Build argument parser with dataset/output options.
    parser = argparse.ArgumentParser(description="Evaluate Advanced RAG Agent with RAGAS and retrieval diagnostics.")
    # Input dataset path.
    parser.add_argument(
        "--csv",
        default="data/artifacts/rag_agent_eval_dataset.csv",
        help="CSV with query and ground_truth columns.",
    )
    # Output CSV path.
    parser.add_argument(
        "--out",
        default="data/artifacts/rag_agent_eval_results.csv",
        help="Output CSV path for row-level RAG predictions.",
    )
    # Parse command-line arguments.
    args = parser.parse_args()

    # Run evaluation and save detailed outputs.
    result = evaluate(Path(args.csv))
    write_predictions(Path(args.out), result["rows"])

    # Print summary report to stdout.
    print("Advanced RAG Agent Evaluation")
    print(f"Dataset: {args.csv}")
    print("")
    print("RAGAS Metrics")
    for name, score in result["ragas"].items():
        print(f"- {name}: {score:.4f}")
    print("")
    print("Operational Metrics")
    print(f"- Retrieval Latency (avg ms): {result['aggregate']['retrieval_latency_ms_avg']}")
    print(f"- Average Similarity: {result['aggregate']['average_similarity']}")
    print(f"- Duplicate Removal (avg count): {result['aggregate']['duplicate_removal_avg']}")
    print("")
    print(f"Detailed row-level output: {args.out}")


# Execute main only when run as a script.
if __name__ == "__main__":
    main()
