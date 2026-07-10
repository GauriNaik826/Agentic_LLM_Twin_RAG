# What: Import ABC (Abstract Base Class) marker from Python's standard library.
# Why: Marks `CleanedDocument` as abstract so it can never be instantiated directly —
#      only the concrete subclasses (CleanedPostDocument, etc.) should be created.
from abc import ABC

# What: Import Optional from typing for fields that may be None.
# Why: Some cleaned document types have fields that are not always present
#      (e.g. an optional image URL on a post), so Optional[str] signals that
#      the field can legitimately be absent without causing a Pydantic validation error.
from typing import Optional

# What: Import UUID4 from Pydantic for type-safe unique identifier fields.
# Why: author_id must reference a UserDocument by its UUID. Using Pydantic's UUID4
#      type ensures the value is validated as a proper v4 UUID on construction,
#      catching bad data early rather than storing a malformed ID.
from pydantic import UUID4

# What: Import VectorBaseDocument — the Qdrant-backed base class for vector-store models.
# Why: CleanedDocument inherits Qdrant persistence methods (bulk_insert, to_point,
#      from_record) and the auto-generated id field from VectorBaseDocument, so this
#      class only needs to declare its own domain fields.
from .base import VectorBaseDocument

# What: Import DataCategory, a StrEnum centralising collection/category name constants.
# Why: Using an enum instead of raw strings (e.g. "posts") prevents typos, makes
#      category names refactor-safe, and is readable throughout the codebase.
from .types import DataCategory


# What: Define the abstract base class shared by all cleaned document types.
# Why: All cleaned documents — regardless of whether they came from a post, article,
#      or repository — share the same core fields. Declaring them once here eliminates
#      duplication across the three concrete subclasses and guarantees a consistent
#      interface for the downstream chunking step.
class CleanedDocument(VectorBaseDocument, ABC):

    # What: The cleaned text content of the document.
    # Why: This is the primary payload produced by the cleaning step — HTML stripped,
    #      whitespace normalised, noise removed. Everything downstream (chunking,
    #      embedding, fine-tuning) operates on this field.
    content: str

    # What: The source platform the document was crawled from (e.g. "medium", "github").
    # Why: Preserved through cleaning so the chunking and embedding steps — and
    #      ultimately the RAG retrieval results — can surface where content originated.
    platform: str

    # What: UUID4 reference to the UserDocument (author) who owns this content.
    # Why: Links every cleaned document back to its author for attribution and
    #      to support the LLM Twin concept of building a per-user knowledge base.
    author_id: UUID4

    # What: The author's full name as a plain string (denormalised from UserDocument).
    # Why: Denormalising avoids a DB lookup at every downstream step. Since the name
    #      is read-only context for display and metadata, storing a copy here is safe.
    author_full_name: str


# What: Concrete cleaned document for social-media post content.
# Why: Posts may carry an optional image URL that other content types don't have.
#      This subclass adds that one extra field while inheriting all shared fields
#      and Qdrant persistence from CleanedDocument / VectorBaseDocument.
class CleanedPostDocument(CleanedDocument):

    # What: Optional URL of an image attached to the post.
    # Why: Some posts are image-heavy; storing the URL keeps it available for
    #      downstream steps or the UI, without making it mandatory for text-only posts.
    image: Optional[str] = None

    # What: Inner Config class read by VectorBaseDocument to configure the Qdrant collection.
    # Why: `name` tells VectorBaseDocument which Qdrant collection to write to;
    #      `category` is used by get_category() for metadata grouping in the ZenML UI;
    #      `use_vector_index = False` means no HNSW vector index is created yet —
    #      vectors are added at the embedding stage, so indexing is deferred until then.
    class Config:
        name = "cleaned_posts"
        category = DataCategory.POSTS
        use_vector_index = False


# What: Concrete cleaned document for long-form article content (e.g. Medium articles).
# Why: Articles always have a canonical URL that is absent from posts and repos.
#      Storing it here means the source link travels through every pipeline stage
#      and can be cited in RAG responses.
class CleanedArticleDocument(CleanedDocument):

    # What: The canonical URL of the article.
    # Why: Provides a direct source link for citation in RAG answers and enables
    #      deduplication if the same article is re-crawled in a later pipeline run.
    link: str

    # What: Qdrant collection config for cleaned article documents.
    # Why: Routes writes to the "cleaned_articles" collection, sets the category
    #      for metadata reporting, and defers vector indexing to the embedding stage.
    class Config:
        name = "cleaned_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


# What: Concrete cleaned document for code-repository content (e.g. GitHub READMEs).
# Why: Repositories have both a human-readable name and a URL — two fields that
#      neither posts nor articles carry. Keeping them here ensures repository
#      identity is preserved all the way through to the final embedded chunks.
class CleanedRepositoryDocument(CleanedDocument):

    # What: The repository name (e.g. "decodingml/llm-twin-course").
    # Why: Human-readable identifier surfaced in RAG context so the user knows
    #      which repository a retrieved snippet came from.
    name: str

    # What: The URL of the repository.
    # Why: Allows direct linking back to the source repo in responses and can be
    #      used for deduplication if the same repo is re-crawled later.
    link: str

    # What: Qdrant collection config for cleaned repository documents.
    # Why: Routes writes to the "cleaned_repositories" collection, sets the category
    #      for metadata reporting, and defers vector indexing to the embedding stage.
    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
