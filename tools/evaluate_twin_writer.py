# Keep annotations deferred for cleaner typing/runtime compatibility.
from __future__ import annotations

# Parse command-line arguments.
import argparse
# Read/write CSV files.
import csv
# Parse JSON text returned by the judge model.
import json
# Compute mean scores across rows.
import statistics
# Work with filesystem paths safely.
from pathlib import Path
# Allow mixed dictionary value types in rows/metrics.
from typing import Any

# Build structured chat prompts for the judge model.
from langchain_core.prompts import ChatPromptTemplate
# OpenAI chat client used for LLM-as-a-Judge scoring.
from langchain_openai import ChatOpenAI

# Twin Writer agent under evaluation.
from llm_engineering.application.agents.twin_writer import TwinWriterAgent
# Runtime settings with model id and API key.
from llm_engineering.settings import settings


# Extract JSON object from raw LLM output text.
def _extract_json(text: str) -> dict[str, Any]:
    # Trim whitespace around model response.
    stripped = text.strip()
    # Remove markdown code fences if the model wrapped JSON in ```json.
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()

    # Locate first opening and last closing curly brace.
    start = stripped.find("{")
    end = stripped.rfind("}")
    # Fail clearly if we cannot find a valid JSON object envelope.
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response does not contain valid JSON object.")

    # Parse only the object slice.
    return json.loads(stripped[start : end + 1])


# Score one generated answer with an LLM judge.
def judge_with_llm(query: str, reference_style: str, generated_answer: str) -> dict[str, float]:
    # OpenAI key is required for the LLM judge path.
    if settings.OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is required for Twin Writer LLM-as-a-Judge evaluation.")

    # Configure judge model to deterministic temperature.
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL_ID,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )

    # Build strict scoring prompt requiring JSON-only output.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict writing evaluator. Return only JSON with integer scores from 1 to 5."
                " Keys must be: style_similarity, grammar, fluency, naturalness.",
            ),
            (
                "human",
                "Reference author style:\n{reference_style}\n\n"
                "Original prompt:\n{query}\n\n"
                "Generated response:\n{generated_answer}\n\n"
                "Compare the generated response with the author's writing style and score:\n"
                "1 (poor) to 5 (excellent) for each metric."
                "Return JSON only.",
            ),
        ]
    )

    # Invoke judge with formatted prompt variables.
    response = llm.invoke(
        prompt.format_messages(
            query=query,
            reference_style=reference_style,
            generated_answer=generated_answer,
        )
    )
    # Parse JSON scores from model output.
    parsed = _extract_json(str(response.content))

    # Required score keys expected from judge output.
    keys = ["style_similarity", "grammar", "fluency", "naturalness"]
    # Normalized score dictionary to return.
    scores: dict[str, float] = {}
    # Validate each key and clamp into [1,5].
    for key in keys:
        value = parsed.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Judge output missing numeric key: {key}")
        clamped = max(1.0, min(5.0, float(value)))
        scores[key] = clamped

    # Return validated score set.
    return scores


# Run full Twin Writer evaluation across CSV dataset.
def evaluate(csv_path: Path) -> dict[str, Any]:
    # Row-level outputs for CSV reporting.
    rows: list[dict[str, Any]] = []

    # Score accumulators for aggregate means.
    style_scores: list[float] = []
    grammar_scores: list[float] = []
    fluency_scores: list[float] = []
    naturalness_scores: list[float] = []

    # Open input dataset and iterate by header.
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Read prompt and reference style profile.
            prompt = row["prompt"]
            reference_style = row["reference_style"]

            # Generate answer from Twin Writer.
            generated = TwinWriterAgent.invoke(prompt)
            # Judge generated answer against reference style.
            judge_scores = judge_with_llm(prompt, reference_style, generated)

            # Collect score values for aggregate metrics.
            style_scores.append(judge_scores["style_similarity"])
            grammar_scores.append(judge_scores["grammar"])
            fluency_scores.append(judge_scores["fluency"])
            naturalness_scores.append(judge_scores["naturalness"])

            # Save row-level record.
            rows.append(
                {
                    "prompt": prompt,
                    "generated_answer": generated,
                    "style_similarity": judge_scores["style_similarity"],
                    "grammar": judge_scores["grammar"],
                    "fluency": judge_scores["fluency"],
                    "naturalness": judge_scores["naturalness"],
                }
            )

    # Compute mean metrics (or 0 if dataset is empty).
    metrics = {
        "style_similarity": round(statistics.mean(style_scores), 2) if style_scores else 0.0,
        "grammar": round(statistics.mean(grammar_scores), 2) if grammar_scores else 0.0,
        "fluency": round(statistics.mean(fluency_scores), 2) if fluency_scores else 0.0,
        "naturalness": round(statistics.mean(naturalness_scores), 2) if naturalness_scores else 0.0,
    }

    # Return both row-level and aggregate outputs.
    return {"rows": rows, "metrics": metrics}


# Write row-level results to output CSV.
def write_results(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    # Ensure parent directory exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Stable column ordering for output file.
    fieldnames = [
        "prompt",
        "generated_answer",
        "style_similarity",
        "grammar",
        "fluency",
        "naturalness",
    ]
    # Write CSV header and rows.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# CLI entrypoint for running the evaluator from terminal.
def main() -> None:
    # Build argument parser.
    parser = argparse.ArgumentParser(description="Evaluate Twin Writer with LLM-as-a-Judge (scores 1-5).")
    # Input dataset argument.
    parser.add_argument(
        "--csv",
        default="data/artifacts/twin_writer_eval_dataset.csv",
        help="CSV with prompt and reference_style columns.",
    )
    # Output CSV argument.
    parser.add_argument(
        "--out",
        default="data/artifacts/twin_writer_eval_results.csv",
        help="Output CSV path for row-level scored results.",
    )
    # Parse CLI values.
    args = parser.parse_args()

    # Run evaluation and persist row-level outputs.
    result = evaluate(Path(args.csv))
    write_results(Path(args.out), result["rows"])

    # Print concise metrics summary.
    metrics = result["metrics"]
    print("Twin Writer Evaluation (LLM-as-a-Judge)")
    print(f"Dataset: {args.csv}")
    print("")
    print("Metrics (1-5)")
    print(f"- Style Similarity: {metrics['style_similarity']:.2f}")
    print(f"- Grammar: {metrics['grammar']:.2f}")
    print(f"- Fluency: {metrics['fluency']:.2f}")
    print(f"- Naturalness: {metrics['naturalness']:.2f}")
    print("")
    print(f"Detailed row-level output: {args.out}")


# Execute main only when this script is run directly.
if __name__ == "__main__":
    main()
