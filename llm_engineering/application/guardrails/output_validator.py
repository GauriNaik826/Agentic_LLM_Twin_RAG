# Keep type hints from being evaluated immediately at runtime.
from __future__ import annotations

# Dataclass helpers for structured validation outputs.
from dataclasses import asdict, dataclass, field
# Generic typing for dynamic metadata/check payloads.
from typing import Any
# Regex support for URL/toxicity/style heuristics.
import re

# Prompt builder used for strict yes/no LLM checks.
from langchain_core.prompts import ChatPromptTemplate
# OpenAI chat client used as a lightweight validator model.
from langchain_openai import ChatOpenAI

# Centralized runtime settings (API key, model ID).
from llm_engineering.settings import settings


# Standard result object returned by validator.
@dataclass
class ValidationResult:
    # Indicates whether all required checks passed.
    passed: bool
    # Numeric quality score in [0, 1].
    confidence: float
    # First failed check label, or None if all checks pass.
    reason: str | None = None
    # Full check-by-check outputs for debugging and telemetry.
    checks: dict[str, Any] = field(default_factory=dict)

    # Convert dataclass into a normal dictionary.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Output validation engine used by Supervisor validator node.
class OutputValidator:
    # URL detector used in citation checks.
    _URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
    # Basic toxicity/unsafe expression patterns.
    _TOXICITY_PATTERNS = [
        re.compile(r"\bhate\b", re.IGNORECASE),
        re.compile(r"\bkill\b", re.IGNORECASE),
        re.compile(r"\bmake\s+(a\s+)?bomb\b", re.IGNORECASE),
        re.compile(r"\bself-harm\b", re.IGNORECASE),
    ]
    # Lower bound for useful response length.
    MIN_WORDS = 20
    # Upper bound to prevent overlong outputs.
    MAX_WORDS = 500

    # Runs a constrained YES/NO LLM judgment.
    def _ask_yes_no(self, system_instruction: str, payload: str) -> bool:
        # If no OpenAI credentials exist, caller falls back to deterministic heuristics.
        if settings.OPENAI_API_KEY is None:
            return False

        # Deterministic evaluator model configuration.
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        # Two-message prompt: strict system rule + payload.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                ("human", payload),
            ]
        )
        # Execute check and normalize response.
        response = llm.invoke(prompt.format_messages())
        answer = response.content.strip().upper()
        # Treat only leading YES as pass.
        return answer.startswith("YES")

    # Checks whether RAG answer is supported by retrieved context.
    def _grounding_check(self, answer: str, retrieved_context: str) -> bool:
        # No context means no grounding support.
        if not retrieved_context.strip():
            return False

        # Fallback heuristic for environments without OpenAI.
        if settings.OPENAI_API_KEY is None:
            # Require at least one overlapping long token between context and answer.
            context_tokens = {token.lower() for token in re.findall(r"[A-Za-z]{5,}", retrieved_context)}
            answer_tokens = {token.lower() for token in re.findall(r"[A-Za-z]{5,}", answer)}
            return len(context_tokens.intersection(answer_tokens)) > 0

        # Build strict grounding prompt payload.
        payload = (
            "Context:\n"
            f"{retrieved_context}\n\n"
            "Answer:\n"
            f"{answer}\n\n"
            "Question:\n"
            "Is the answer fully supported by the context?\n\n"
            "Respond only:\nYES\nNO"
        )
        # Use LLM as binary grounding judge.
        return self._ask_yes_no(
            "You are a strict grounding checker. Return only YES or NO.",
            payload,
        )

    # Validates presence of source/citation markers.
    def _citation_check(self, answer: str) -> bool:
        # URL-based evidence marker.
        has_url = bool(self._URL_PATTERN.search(answer))
        # Alternative citation markers used in generated text.
        has_source_marker = "source:" in answer.lower() or "sources:" in answer.lower() or "[1]" in answer
        return has_url or has_source_marker

    # Checks if answer style matches expected twin-writer tone.
    def _style_check(self, query: str, answer: str) -> bool:
        # Heuristic fallback when no evaluator LLM is available.
        if settings.OPENAI_API_KEY is None:
            # Look for first-person and practical language hints.
            return bool(re.search(r"\bI\b|\bwe\b|\bpractical\b|\bexample\b", answer, re.IGNORECASE))

        # Build style-check prompt payload.
        payload = (
            "User Query:\n"
            f"{query}\n\n"
            "Answer:\n"
            f"{answer}\n\n"
            "Question:\n"
            "Does this match an engaging technical author style (clear, practical, concise)?\n\n"
            "Respond only:\nYES\nNO"
        )
        # Use LLM as binary style judge.
        return self._ask_yes_no(
            "You are a style validator. Return only YES or NO.",
            payload,
        )

    # Primary validation entrypoint used by supervisor validator node.
    def validate(
        self,
        *,
        route: str,
        query: str,
        answer: str,
        retrieved_context: str = "",
        web_results: list[dict[str, Any]] | None = None,
    ) -> ValidationResult:
        # Normalize optional container.
        web_results = web_results or []
        # Per-check result bucket.
        checks: dict[str, Any] = {}

        # 1) Empty Response
        # Immediate failure for blank outputs.
        if not answer.strip():
            checks["empty_answer"] = True
            return ValidationResult(passed=False, confidence=0.0, reason="empty_answer", checks=checks)
        checks["empty_answer"] = False

        # 2) Grounding for RAG
        # Grounding is enforced only for RAG route.
        if route == "rag":
            checks["grounding_ok"] = self._grounding_check(answer, retrieved_context)
        else:
            checks["grounding_ok"] = True

        # 3) Citation check for Web and RAG
        # Citation requirement applies to factual routes.
        if route in {"web", "rag"}:
            checks["citation_ok"] = self._citation_check(answer)
        else:
            checks["citation_ok"] = True

        # 4) Style check for Twin writer
        # Style validation applies only to twin-writer route.
        if route == "twin_writer":
            checks["style_ok"] = self._style_check(query, answer)
        else:
            checks["style_ok"] = True

        # Toxicity check
        # Collect matched toxicity regex patterns.
        toxicity_hits = [pattern.pattern for pattern in self._TOXICITY_PATTERNS if pattern.search(answer)]
        checks["toxicity_ok"] = len(toxicity_hits) == 0
        checks["toxicity_hits"] = toxicity_hits

        # Length check
        # Compute simple token proxy using whitespace words.
        word_count = len(answer.split())
        checks["length_ok"] = self.MIN_WORDS <= word_count <= self.MAX_WORDS
        checks["word_count"] = word_count

        # Lightweight hallucination check based on support indicators.
        # Reuse route-specific support checks as hallucination proxy.
        if route == "rag":
            checks["hallucination_ok"] = checks["grounding_ok"]
        elif route == "web":
            checks["hallucination_ok"] = checks["citation_ok"] and (len(web_results) > 0 or checks["citation_ok"])
        else:
            checks["hallucination_ok"] = True

        # Gather names of failed critical checks.
        failed_checks = [
            name
            for name in ["grounding_ok", "citation_ok", "style_ok", "toxicity_ok", "length_ok", "hallucination_ok"]
            if checks.get(name) is False
        ]

        # Penalty weights used to convert failures into confidence score.
        penalty_map = {
            "grounding_ok": 0.25,
            "citation_ok": 0.20,
            "style_ok": 0.10,
            "toxicity_ok": 0.30,
            "length_ok": 0.10,
            "hallucination_ok": 0.20,
        }
        # Start with high confidence, subtract penalties for each failed check.
        confidence = 0.95 - sum(penalty_map.get(name, 0.0) for name in failed_checks)
        # Clamp confidence into [0,1] and round for readability.
        confidence = max(0.0, min(1.0, round(confidence, 2)))

        # First failed check becomes primary reason.
        reason = failed_checks[0] if failed_checks else None
        # Return structured validation output.
        return ValidationResult(
            passed=len(failed_checks) == 0,
            confidence=confidence,
            reason=reason,
            checks=checks,
        )