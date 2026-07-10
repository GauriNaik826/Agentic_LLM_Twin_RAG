# Keep annotations deferred for better typing compatibility.
from __future__ import annotations

# Parse CLI arguments.
import argparse
# Read/write CSV files.
import csv
# Parse JSON returned by judge model.
import json
# Handle filesystem paths safely.
from pathlib import Path
# Allow mixed dictionary value types in outputs.
from typing import Any

# Import the end-to-end supervisor entrypoint being evaluated.
from llm_engineering.application.orchestration.supervisor import Supervisor
# Import runtime settings, including optional OpenAI API key.
from llm_engineering.settings import settings


# Allowed expected behavior labels in dataset.
BEHAVIORS = {"rag", "web", "twin_writer"}


# Divide safely to avoid division-by-zero in aggregate metrics.
def safe_div(numerator: float, denominator: float) -> float:
    # Return 0 when denominator has no samples.
    if denominator == 0:
        return 0.0
    # Normal division path.
    return numerator / denominator


# Constrain judge scores to the expected 1-5 range.
def clamp_score(value: float) -> float:
    # Clamp the number into [1.0, 5.0].
    return max(1.0, min(5.0, value))


# Extract a JSON object from raw judge output text.
def _extract_json(text: str) -> dict[str, Any]:
    # Trim leading/trailing whitespace.
    raw = text.strip()
    # Remove markdown code fences if model wrapped JSON in ```json blocks.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    # Locate outer JSON object boundaries.
    start = raw.find("{")
    end = raw.rfind("}")
    # Raise clear error if object boundaries are not found.
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge did not return JSON object.")
    # Parse substring into Python dictionary.
    return json.loads(raw[start : end + 1])


# Judge final answer quality against expected behavior using LLM or heuristic fallback.
def llm_judge(query: str, expected_behavior: str, answer: str) -> dict[str, Any]:
    # If OpenAI is unavailable, use deterministic heuristic scoring.
    if settings.OPENAI_API_KEY is None:
        # Heuristic fallback when judge model is unavailable.
        answer_lower = answer.lower()
        task_success = False
        # Web tasks are treated as successful if answer contains source/citation hints.
        if expected_behavior == "web":
            task_success = "http://" in answer_lower or "https://" in answer_lower or "source" in answer_lower
        # Twin writer tasks are treated as successful on simple first-person writing cues.
        elif expected_behavior == "twin_writer":
            task_success = "i" in answer_lower or "we" in answer_lower
        # RAG-like tasks use a simple minimum length proxy.
        else:
            task_success = len(answer.split()) >= 20

        # Return heuristic judgments with fixed score policy.
        return {
            "task_success": task_success,
            "user_satisfaction": 3.0 if task_success else 2.0,
            "groundedness": 3.0 if expected_behavior != "web" else (4.0 if task_success else 2.0),
            "style_consistency": 3.0 if expected_behavior == "twin_writer" else 2.5,
            "reason": "heuristic_judge_no_openai",
        }

    # Import LangChain judge components lazily to avoid hard dependency in heuristic mode.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Build strict prompt that asks for JSON-only evaluation output.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an end-to-end evaluator. Evaluate only user intent vs final answer quality. "
                "Ignore internal architecture details. Return only JSON.",
            ),
            (
                "human",
                "Query:\n{query}\n\n"
                "Expected behavior (one of rag/web/twin_writer):\n{expected_behavior}\n\n"
                "Final answer:\n{answer}\n\n"
                "Return JSON with keys:\n"
                "task_success (boolean),\n"
                "user_satisfaction (number 1-5),\n"
                "groundedness (number 1-5),\n"
                "style_consistency (number 1-5),\n"
                "reason (short string).",
            ),
        ]
    )

    # Create deterministic judge model configuration.
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL_ID,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )

    # Run evaluation prompt against the final answer.
    response = llm.invoke(
        prompt.format_messages(
            query=query,
            expected_behavior=expected_behavior,
            answer=answer,
        )
    )
    # Parse JSON content from model response.
    parsed = _extract_json(str(response.content))

    # Read and normalize returned fields.
    task_success = bool(parsed.get("task_success", False))
    user_satisfaction = clamp_score(float(parsed.get("user_satisfaction", 1.0)))
    groundedness = clamp_score(float(parsed.get("groundedness", 1.0)))
    style_consistency = clamp_score(float(parsed.get("style_consistency", 1.0)))
    reason = str(parsed.get("reason", ""))

    # Return normalized judgment payload.
    return {
        "task_success": task_success,
        "user_satisfaction": user_satisfaction,
        "groundedness": groundedness,
        "style_consistency": style_consistency,
        "reason": reason,
    }


# Run full end-to-end evaluation over dataset rows.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Instantiate supervisor once and reuse for all queries.
    supervisor = Supervisor()

    # Row-level output container.
    rows: list[dict[str, Any]] = []
    # Aggregate counters and score accumulators.
    total = 0
    success_count = 0
    user_satisfaction_scores: list[float] = []
    groundedness_scores: list[float] = []
    style_scores: list[float] = []

    # Read evaluation dataset and process each query.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Normalize input fields.
            query = row["query"].strip()
            expected_behavior = row["expected_behavior"].strip().lower().replace(" ", "_")
            # Validate dataset label quality.
            if expected_behavior not in BEHAVIORS:
                raise ValueError(f"Invalid expected_behavior: {expected_behavior}")

            # Generate final answer through end-to-end supervisor pipeline.
            answer = supervisor.invoke(query)
            # Evaluate final answer with LLM/heuristic judge.
            judgment = llm_judge(query, expected_behavior, answer)

            # Update aggregate task-success counters.
            total += 1
            if judgment["task_success"]:
                success_count += 1

            # Collect numeric scores for averaging.
            user_satisfaction_scores.append(float(judgment["user_satisfaction"]))
            groundedness_scores.append(float(judgment["groundedness"]))
            style_scores.append(float(judgment["style_consistency"]))

            # Save row-level evaluated output.
            rows.append(
                {
                    "query": query,
                    "expected_behavior": expected_behavior,
                    "answer": answer,
                    "task_success": judgment["task_success"],
                    "user_satisfaction": round(float(judgment["user_satisfaction"]), 2),
                    "groundedness": round(float(judgment["groundedness"]), 2),
                    "style_consistency": round(float(judgment["style_consistency"]), 2),
                    "reason": judgment["reason"],
                }
            )

    # Compute aggregate Step 6 metrics.
    metrics = {
        "task_success_rate": round(safe_div(success_count, total), 4),
        "user_satisfaction": round(safe_div(sum(user_satisfaction_scores), len(user_satisfaction_scores)), 2),
        "groundedness": round(safe_div(sum(groundedness_scores), len(groundedness_scores)), 2),
        "style_consistency": round(safe_div(sum(style_scores), len(style_scores)), 2),
        "total_rows": total,
    }

    # Return row-level results and summary metrics.
    return {"rows": rows, "metrics": metrics}


# Write row-level evaluation results to CSV file.
def write_results(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Ensure output directory exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Stable output column order.
    fieldnames = [
        "query",
        "expected_behavior",
        "answer",
        "task_success",
        "user_satisfaction",
        "groundedness",
        "style_consistency",
        "reason",
    ]
    # Write header and all rows.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# CLI entrypoint for running end-to-end evaluation from terminal.
def main() -> None:
    # Build argument parser.
    parser = argparse.ArgumentParser(description="End-to-end user-to-answer evaluation.")
    # Input dataset option.
    parser.add_argument(
        "--csv",
        default="data/artifacts/end_to_end_eval_dataset.csv",
        help="Input CSV with query and expected_behavior columns.",
    )
    # Output CSV option.
    parser.add_argument(
        "--out",
        default="data/artifacts/end_to_end_eval_results.csv",
        help="Output CSV path for row-level end-to-end evaluation.",
    )
    # Parse CLI arguments.
    args = parser.parse_args()

    # Execute evaluation and save row-level output.
    result = evaluate(Path(args.csv))
    write_results(Path(args.out), result["rows"])

    # Print concise summary metrics.
    metrics = result["metrics"]
    print("End-to-End Evaluation")
    print(f"Dataset: {args.csv}")
    print("")
    print("Metrics")
    print(f"- Task Success Rate: {metrics['task_success_rate']:.2%}")
    print(f"- User Satisfaction (1-5): {metrics['user_satisfaction']:.2f}")
    print(f"- Groundedness (1-5): {metrics['groundedness']:.2f}")
    print(f"- Style Consistency (1-5): {metrics['style_consistency']:.2f}")
    print("")
    print(f"Total Rows: {metrics['total_rows']}")
    print(f"Detailed row-level output: {args.out}")


# Run main only when this file is executed directly.
if __name__ == "__main__":
    main()
