# What: Import ABC (Abstract Base Class) marker from Python's standard library.
# Why: Marks `Chunk` as abstract so it can never be instantiated directly —
#      only concrete subclasses (PostChunk, ArticleChunk, RepositoryChunk) can be created.
from abc import ABC

# What: Import Optional from typing for fields that may be None.
# Why: Some chunk types (e.g. posts) have an optional image URL — Optional[str]
#      makes that explicit and lets Pydantic skip validation when the value is absent.
from typing import Optional

# What: Import UUID4 and Field from Pydantic.
# Why: UUID4 enforces type-safe universally unique IDs for author and document
#      references; Field(default_factory=dict) safely initialises the metadata
#      dict to a new empty dict per instance (avoids the shared-mutable-default pitfall).
from pydantic import UUID4, Field

# What: Import VectorBaseDocument — the Qdrant-backed base class for all vector-store models.
# Why: Chunk inherits persistence methods (bulk_insert, from_record, to_point) and
#      the auto-generated UUID id field from VectorBaseDocument, so the class itself
#      only needs to declare its own fields.
from llm_engineering.domain.base import VectorBaseDocument

# What: Import DataCategory, a StrEnum centralising collection/category name constants.
# Why: Using an enum instead of raw strings (e.g. "posts") prevents typos and makes
#      category names refactor-safe everywhere they are referenced.
from llm_engineering.domain.types import DataCategory


# What: Define the abstract base class for all text chunks produced by the chunking step.
# Why: All chunk types share the same core fields (content, platform, author, etc.).
#      Putting them here once avoids duplication across PostChunk, ArticleChunk, and
#      RepositoryChunk, and guarantees every chunk can be stored in Qdrant via VectorBaseDocument.
class Chunk(VectorBaseDocument, ABC):

    # What: The actual text content of this chunk.
    # Why: This is the payload that gets embedded into a vector and later retrieved
    #      by the RAG system to answer queries. It is the core field of every chunk.
    content: str

    # What: The source platform this chunk originated from (e.g. "medium", "github").
    # Why: Preserved at chunk level so that during retrieval we can surface *where*
    #      the content came from — important for attribution and debugging.
    platform: str

    # What: UUID4 reference to the parent cleaned document this chunk was derived from.
    # Why: Maintains lineage — we can always trace a chunk back to the exact cleaned
    #      document (and in turn, the raw document) it came from.
    document_id: UUID4

    # What: UUID4 reference to the author (UserDocument) who produced this content.
    # Why: Enables author-scoped retrieval — e.g. "find all chunks written by this author" —
    #      and supports the LLM Twin concept of personalising the model per user.
    author_id: UUID4

    # What: The author's full name as a plain string (denormalised from UserDocument).
    # Why: Denormalising the name avoids a join/lookup at retrieval time and makes
    #      the chunk self-contained for display in RAG context windows.
    author_full_name: str

    # What: A flexible dict for any extra metadata specific to the chunk or its source.
    # Why: Keeps the base class extensible — subclasses or the chunking step can attach
    #      source URL, chunk index, token count, etc., without changing the schema.
    #      default_factory=dict creates a fresh empty dict for every instance, preventing
    #      the classic "all instances sharing the same mutable default" bug.
    metadata: dict = Field(default_factory=dict)


# What: Concrete chunk type for social-media post content.
# Why: Posts may include an image URL alongside text; this subclass adds that optional
#      field while inheriting all shared Chunk fields and Qdrant persistence from the base.
class PostChunk(Chunk):

    # What: Optional URL of an image attached to the post.
    # Why: Some posts are image-heavy; storing the URL at chunk level lets downstream
    #      steps or the UI render the image alongside the retrieved text context.
    image: Optional[str] = None

    # What: Inner Config class that tells VectorBaseDocument which category this type belongs to.
    # Why: VectorBaseDocument.get_category() reads Config.category to route the chunk
    #      to the correct Qdrant collection and to populate per-category metadata stats
    #      in the ZenML pipeline dashboard.
    class Config:
        category = DataCategory.POSTS


# What: Concrete chunk type for long-form article content (e.g. Medium posts).
# Why: Articles always have a canonical URL; storing it at chunk level allows the
#      RAG system to cite the source article when returning retrieved context.
class ArticleChunk(Chunk):

    # What: The URL of the article this chunk was extracted from.
    # Why: Provides a direct source link for citation in RAG responses and enables
    #      deduplication checks if the same article is re-crawled.
    link: str

    # What: Category config routing this chunk type to the ARTICLES collection in Qdrant.
    # Why: Keeps article chunks in their own collection, enabling category-filtered
    #      vector searches (e.g. "search only in articles").
    class Config:
        category = DataCategory.ARTICLES


# What: Concrete chunk type for code-repository content (e.g. GitHub README, code files).
# Why: Repositories have both a name and a URL, which are meaningful metadata for
#      code-related queries — e.g. the RAG system can cite the exact repo and file.
class RepositoryChunk(Chunk):

    # What: The repository name (e.g. "decodingml/llm-twin-course").
    # Why: Human-readable identifier for the source repo, surfaced in RAG context
    #      so the user knows the retrieved code snippet's origin.
    name: str

    # What: The URL of the repository.
    # Why: Allows direct linking back to the source repo in responses, and can be
    #      used to deduplicate if the same repo is re-crawled in a later pipeline run.
    link: str

    # What: Category config routing this chunk type to the REPOSITORIES collection in Qdrant.
    # Why: Separates code content from article and post content in Qdrant, enabling
    #      targeted retrieval (e.g. "only search code repositories for this query").
    class Config:
        category = DataCategory.REPOSITORIES
