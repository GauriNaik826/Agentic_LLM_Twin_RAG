"""
documents.py — Domain document models for the LLM Twin data warehouse
----------------------------------------------------------------------

WHY THIS FILE EXISTS
--------------------
The LLM Twin ingestion pipeline collects raw data from multiple external
sources (GitHub, Medium, LinkedIn, generic articles) and a local user
registry. Each source produces structurally different data, but all of it
must be persisted to and retrieved from MongoDB in a consistent, type-safe
way.

This file defines the *domain document layer* — Pydantic models that:
1. Declare the schema (fields and types) for every MongoDB collection.
2. Inherit persistence helpers (save, find, bulk_insert…) from
   `NoSQLBaseDocument` so every model works with MongoDB out of the box.
3. Identify the target MongoDB collection via an inner `Settings.name`
   attribute (resolved by `NoSQLBaseDocument.get_collection_name()`).

HOW THE LAYERS FIT TOGETHER
----------------------------
  External source (GitHub / Medium / LinkedIn / any article)
        ↓ (crawled by a specific crawler)
  Concrete Document model  ← this file
        ↓ (.save())
  MongoDB collection       ← named by Settings.name / DataCategory
        ↓ (queried during feature engineering / RAG)
  Cleaned / embedded chunks for LLM fine-tuning and retrieval

DESIGN PRINCIPLES
-----------------
- Each model is a plain Pydantic class — easy to validate, serialise, and
  test without a running database.
- The shared `Document` base class captures fields common to all crawled
  content (content, platform, author), eliminating duplication.
- `UserDocument` is intentionally separate because it represents a system
  entity (the person whose data is being twinned) rather than crawled content.
"""

from abc import ABC          # Used to mark `Document` as abstract — it cannot be
                             # instantiated directly, only its concrete subclasses can.
from typing import Optional  # Allows fields to be None (e.g., optional image URL)

from pydantic import UUID4, Field  # UUID4: type-safe universally unique IDs;
                                   # Field: customise field metadata (aliases, defaults)

# NoSQLBaseDocument provides the MongoDB persistence layer:
# .save(), .find(), .bulk_find(), .get_or_create(), .bulk_insert(),
# and the id field (UUID4). All domain documents inherit from it.
from .base import NoSQLBaseDocument

# DataCategory is a StrEnum that centralises collection name constants.
# Using an enum instead of bare strings prevents typos and makes
# collection names refactor-safe across the whole codebase.
from .types import DataCategory


# ---------------------------------------------------------------------------
# UserDocument
# ---------------------------------------------------------------------------

class UserDocument(NoSQLBaseDocument):
    """Represents a user (the person whose online data is being twinned).

    This is the only document that is NOT crawled content — it is created
    by the ETL pipeline's `get_or_create_user` step from a config file.
    It is stored so that author metadata (id and full_name) can be
    attached to every crawled document for traceability and attribution.
    """

    first_name: str  # Given name — stored separately so full_name can be
                     # reconstructed and to allow future search/filter by first name
    last_name: str   # Family name

    class Settings:
        # Maps this model to the MongoDB "users" collection.
        # `NoSQLBaseDocument.get_collection_name()` reads this value.
        name = "users"

    @property
    def full_name(self):
        # Convenience property used throughout the pipeline when attaching
        # author attribution to crawled documents. Defined here once so
        # callers never have to concatenate first/last manually.
        return f"{self.first_name} {self.last_name}"


# ---------------------------------------------------------------------------
# Document — shared abstract base for all crawled content
# ---------------------------------------------------------------------------

class Document(NoSQLBaseDocument, ABC):
    """Abstract base class for all crawled content documents.

    WHY THIS EXISTS
    ---------------
    GitHub repos, LinkedIn posts, Medium articles, and generic articles all
    share a common set of fields: the content payload, the source platform,
    and the author who owns the data. Centralising these fields here means:
    - Concrete subclasses only declare their own *unique* fields.
    - Any code that handles "any crawled document" can type-hint against
      `Document` and access these fields safely.

    This class is marked ABC so it cannot be instantiated directly — you
    must use a concrete subclass (RepositoryDocument, PostDocument, etc.).
    """

    # The main extracted payload stored as a flexible dict. Using `dict`
    # (rather than a typed sub-model) keeps the schema open to the different
    # structures each crawler produces (e.g., file-path→content for repos,
    # title/body/language for articles) without forcing a one-size-fits-all schema.
    content: dict

    # The source domain/platform where the content was crawled from
    # (e.g., "github", "medium.com", "linkedin.com"). Stored so downstream
    # pipelines can filter or weight data by source.
    platform: str

    # UUID of the UserDocument whose data this content belongs to.
    # Stored alongside full_name for efficient attribution without a join.
    # `alias="author_id"` ensures field name matches when serialising to/from
    # MongoDB documents (where the key is "author_id").
    author_id: UUID4 = Field(alias="author_id")

    # Human-readable author name stored redundantly alongside author_id.
    # Redundancy is intentional: it avoids a round-trip lookup to `users`
    # collection every time we need to display or log author attribution.
    author_full_name: str = Field(alias="author_full_name")


# ---------------------------------------------------------------------------
# RepositoryDocument
# ---------------------------------------------------------------------------

class RepositoryDocument(Document):
    """Stores a cloned GitHub repository as a structured document.

    The `content` field (inherited from Document) holds a dict mapping
    relative file paths to file contents, as built by `GithubCrawler`.
    This preserves the repository's directory hierarchy for downstream code
    analysis, chunking, and RAG retrieval.
    """

    # Human-readable repository name extracted from the GitHub URL
    # (e.g., "llm-twin"). Useful for display and filtering.
    name: str

    # The original GitHub URL. Serves as the deduplication key:
    # `GithubCrawler` queries by link before re-cloning.
    link: str

    class Settings:
        # Routes persistence to the "repositories" MongoDB collection.
        # DataCategory.REPOSITORIES == "repositories" (StrEnum).
        name = DataCategory.REPOSITORIES


# ---------------------------------------------------------------------------
# PostDocument
# ---------------------------------------------------------------------------

class PostDocument(Document):
    """Stores a social-media post (e.g., a LinkedIn post).

    Posts can optionally include an image URL and a direct link. Both are
    optional because not every post has them, and making them Optional
    avoids validation errors when the crawler cannot extract these fields.
    """

    # URL of an associated image, if present in the post.
    # Optional because many text-only posts have no image.
    image: Optional[str] = None

    # Direct URL to the post. Optional because some platforms don't
    # expose a canonical permalink for every post.
    link: str | None = None

    class Settings:
        # Routes persistence to the "posts" MongoDB collection.
        name = DataCategory.POSTS


# ---------------------------------------------------------------------------
# ArticleDocument
# ---------------------------------------------------------------------------

class ArticleDocument(Document):
    """Stores a scraped web article (Medium post, blog entry, etc.).

    The `content` field (inherited) holds the structured extraction dict
    built by `CustomArticleCrawler` or `MediumCrawler`, containing:
    Title, Subtitle, Content (plain text), and language.
    """

    # The canonical URL of the article. Required (not Optional) because
    # articles without a stable URL cannot be deduplicated on re-run.
    # `CustomArticleCrawler` uses this as the deduplication key via
    # `self.model.find(link=link)` before fetching the page.
    link: str

    class Settings:
        # Routes persistence to the "articles" MongoDB collection.
        name = DataCategory.ARTICLES
