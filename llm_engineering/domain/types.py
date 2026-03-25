"""Shared domain-level enum constants.

This module defines canonical string categories used across the project,
primarily for naming MongoDB collections and artifact buckets consistently.
Keeping them centralized prevents typo-driven bugs (e.g., "article" vs
"articles") and makes renaming safe.
"""

from enum import StrEnum


class DataCategory(StrEnum):
        """Canonical categories used by documents, datasets, and pipelines.

        Why `StrEnum`?
        - Members behave like both enums and strings.
        - This gives type-safety in Python code while still passing directly to
            APIs that expect plain strings (e.g., MongoDB collection names).
        """

        # Prompt artifacts used by prompting/inference utilities.
        PROMPT = "prompt"

        # User query artifacts used by retrieval/evaluation flows.
        QUERIES = "queries"

        # Per-example intermediate samples for instruction-style dataset creation.
        INSTRUCT_DATASET_SAMPLES = "instruct_dataset_samples"

        # Final instruction-tuning dataset artifact/collection.
        INSTRUCT_DATASET = "instruct_dataset"

        # Per-example intermediate samples for preference dataset generation.
        PREFERENCE_DATASET_SAMPLES = "preference_dataset_samples"

        # Final preference dataset artifact/collection.
        PREFERENCE_DATASET = "preference_dataset"

        # Social post documents (e.g., LinkedIn) ingested by crawlers.
        POSTS = "posts"

        # Article documents ingested by generic/Medium crawlers.
        ARTICLES = "articles"

        # Code repository documents ingested by GitHub crawler.
        REPOSITORIES = "repositories"
