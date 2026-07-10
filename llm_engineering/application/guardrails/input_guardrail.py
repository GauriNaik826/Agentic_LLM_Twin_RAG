# What: Postpone runtime evaluation of type annotations.
# Why: Keeps typing flexible and avoids forward-reference issues.
from __future__ import annotations

# What: Import regex engine.
# Why: Used for detection/masking patterns.
import re
# What: Import Unicode normalization helpers.
# Why: Normalizes visually similar characters before guardrail checks.
import unicodedata
# What: Import generic typing alias.
# Why: Metadata dictionary stores mixed value types.
from typing import Any


# What: Exception type for prompt injection detection.
# Why: Lets supervisor handle this failure path explicitly.
class UnsafePromptException(Exception):
    """Raised when prompt-injection patterns are detected."""


# What: Exception type for unsupported/harmful requests.
# Why: Allows early block before routing/execution.
class UnsupportedRequestException(Exception):
    """Raised when the request is outside allowed behavior."""


# What: Input guardrail component.
# Why: First-stage sanitizer and safety gate before supervisor routing.
class InputGuardrail:
    # What: Safe message returned for injection attempts.
    # Why: Avoids leaking internals while explaining refusal.
    SAFE_INJECTION_RESPONSE = "I cannot comply with instruction-override or prompt-injection requests."
    # What: Safe message returned for unsupported/harmful intent.
    # Why: Keeps response policy consistent and minimal.
    SAFE_UNSUPPORTED_RESPONSE = "I cannot help with that request."

    # What: PII detection patterns by type.
    # Why: Enables redaction and metadata flags for sensitive user data.
    _PII_PATTERNS = {
        # What: Email pattern.
        # Why: Masks email addresses such as user@example.com.
        "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        # What: Phone pattern.
        # Why: Masks common phone formats (with optional country code/separators).
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
        # What: US SSN pattern.
        # Why: Masks social security numbers in ###-##-#### format.
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # What: Credit card pattern.
        # Why: Masks common 13-19 digit card-like sequences.
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        # What: API key pattern.
        # Why: Masks common secret token prefixes.
        "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{16,}|AIza[0-9A-Za-z\-_]{20,})\b"),
    }

    # What: Prompt-injection intent patterns.
    # Why: Blocks instructions attempting to override system behavior.
    _INJECTION_PATTERNS = [
        re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b", re.IGNORECASE),
        re.compile(r"\breveal\s+(the\s+)?system\s+prompt\b", re.IGNORECASE),
        re.compile(r"\bforget\s+everything\b", re.IGNORECASE),
        re.compile(r"\bdo\s+not\s+follow\s+your\s+rules\b", re.IGNORECASE),
    ]

    # What: Unsupported/harmful request patterns.
    # Why: Refuses abusive cyber misuse requests early.
    _UNSUPPORTED_PATTERNS = [
        re.compile(r"\bhack\b", re.IGNORECASE),
        re.compile(r"\bphish(?:ing)?\b", re.IGNORECASE),
        re.compile(r"\bsteal\b", re.IGNORECASE),
        re.compile(r"\bmalware\b", re.IGNORECASE),
        re.compile(r"\bransomware\b", re.IGNORECASE),
    ]

    # What: Normalize raw user text.
    # Why: Standardizes input before safety checks and routing.
    def normalize_query(self, query: str) -> str:
        # What: Normalize Unicode to compatibility form.
        # Why: Prevents evasion via weird unicode variants.
        cleaned = unicodedata.normalize("NFKC", query)
        # What: Remove selected markdown/control symbols.
        # Why: Reduces prompt noise and formatting artifacts.
        cleaned = re.sub(r"[`*_>#~]", " ", cleaned)
        # What: Collapse repeated whitespace.
        # Why: Produces stable, compact normalized query.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # What: Return normalized text.
        # Why: Downstream checks operate on this cleaned form.
        return cleaned

    # What: Replace detected PII with typed placeholders.
    # Why: Protects sensitive user data from propagation.
    def _mask_pii(self, query: str) -> tuple[str, list[str]]:
        # What: Start with unmodified input.
        # Why: Apply replacements progressively.
        masked = query
        # What: Track which PII types were found.
        # Why: Exposed via metadata for observability/policy use.
        detected_types: list[str] = []

        # What: Apply each PII regex in sequence.
        # Why: Supports multiple sensitive data categories.
        for pii_type, pattern in self._PII_PATTERNS.items():
            # What: Substitute matches with typed token.
            # Why: Preserve structure while redacting secrets.
            new_masked = pattern.sub(f"[{pii_type}]", masked)
            # What: Detect whether replacements happened.
            # Why: Record PII presence accurately.
            if new_masked != masked:
                detected_types.append(pii_type)
                masked = new_masked

        # What: Return redacted text and detected type list.
        # Why: Caller needs both transformed query and metadata.
        return masked, detected_types

    # What: Determine whether query contains injection patterns.
    # Why: Blocks instruction hijacking attempts.
    def _contains_prompt_injection(self, query: str) -> bool:
        # What: True if any injection regex matches.
        # Why: Simple, conservative policy gate.
        return any(pattern.search(query) for pattern in self._INJECTION_PATTERNS)

    # What: Determine whether query contains unsupported intent.
    # Why: Refuses harmful request classes before any agent execution.
    def _contains_unsupported_request(self, query: str) -> bool:
        # What: True if any unsupported regex matches.
        # Why: Fast policy decision path.
        return any(pattern.search(query) for pattern in self._UNSUPPORTED_PATTERNS)

    # What: Main guardrail processing pipeline.
    # Why: Produces safe normalized/redacted query + metadata or raises block exceptions.
    def process(self, query: str) -> tuple[str, dict[str, Any]]:
        # What: Normalize user input first.
        # Why: Ensures consistent detection behavior.
        normalized_query = self.normalize_query(query)

        # What: Block prompt injection attempts.
        # Why: Prevents supervisor/agents from following override instructions.
        if self._contains_prompt_injection(normalized_query):
            raise UnsafePromptException(self.SAFE_INJECTION_RESPONSE)

        # What: Block unsupported harmful requests.
        # Why: Enforces high-level safety policy.
        if self._contains_unsupported_request(normalized_query):
            raise UnsupportedRequestException(self.SAFE_UNSUPPORTED_RESPONSE)

        # What: Redact PII from normalized query.
        # Why: Avoid passing sensitive data deeper into the system.
        redacted_query, pii_types = self._mask_pii(normalized_query)

        # What: Build guardrail metadata payload.
        # Why: Allows supervisor/telemetry to track PII handling outcomes.
        metadata: dict[str, Any] = {
            "pii_detected": len(pii_types) > 0,
            "pii_types": pii_types,
        }

        # What: Return safe query and metadata.
        # Why: Downstream routing/execution uses this cleaned input.
        return redacted_query, metadata