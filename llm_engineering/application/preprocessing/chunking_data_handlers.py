# What: Import hashlib for deterministic ID generation from chunk content.
# Why: Rather than generating random UUIDs, we hash the chunk text with MD5 so the
#      same text always produces the same chunk ID. This makes re-runs idempotent —
#      re-processing the same document does not create duplicate chunks in Qdrant.
import hashlib

# What: Import ABC and abstractmethod to define the abstract base handler.
# Why: ABC enforces that ChunkingDataHandler is never instantiated directly;
#      abstractmethod forces every concrete subclass to implement chunk(),
#      guaranteeing a consistent interface regardless of document type.
from abc import ABC, abstractmethod

# What: Import Generic and TypeVar for type-safe generics.
# Why: Generic[CleanedDocumentT, ChunkT] lets each concrete handler declare exactly
#      which input/output types it works with (e.g. CleanedPostDocument → PostChunk),
#      giving static type checkers full visibility without losing the shared base class.
from typing import Generic, TypeVar

# What: Import UUID for converting MD5 hex strings into proper UUID objects.
# Why: Qdrant and the domain models require UUID4 IDs; UUID(..., version=4) wraps
#      the 32-character MD5 hex digest into the correct UUID format.
from uuid import UUID

# What: Import the concrete chunk domain models produced by each handler.
# Why: Each handler creates instances of the appropriate chunk type so the output
#      is a strongly typed list (e.g. list[PostChunk]) rather than a plain dict.
from llm_engineering.domain.chunks import ArticleChunk, Chunk, PostChunk, RepositoryChunk

# What: Import the cleaned document domain models consumed by each handler.
# Why: Each handler's chunk() method accepts a specific cleaned document type;
#      importing them gives us precise type annotations and IDE/type-checker support.
from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

# What: Import the actual text-splitting functions that do the chunking work.
# Why: chunk_text splits by character count with overlap (used for posts and repos);
#      chunk_article uses semantic/sentence-aware splitting tuned for long articles.
#      Separating these utilities from the handlers keeps each file single-responsibility.
from .operations import chunk_article, chunk_text

# What: TypeVar for the input cleaned document type, bound to CleanedDocument.
# Why: Allows ChunkingDataHandler[CleanedPostDocument, PostChunk] to be valid while
#      still enforcing that the input must be a CleanedDocument subclass.
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)

# What: TypeVar for the output chunk type, bound to Chunk.
# Why: Pairs with CleanedDocumentT so each handler's input/output types are
#      linked — e.g. CleanedArticleDocument always produces ArticleChunk objects.
ChunkT = TypeVar("ChunkT", bound=Chunk)


# What: Abstract base class that all concrete chunking handlers must inherit from.
# Why: Defines the shared contract (metadata property + chunk() method) so the
#      ChunkingDispatcher can call any handler through a unified interface, applying
#      the Strategy behavioural pattern — the algorithm (chunking strategy) is
#      swapped at runtime depending on the document type.
class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    # What: Default metadata property providing fallback chunk_size and chunk_overlap values.
    # Why: Subclasses override this to tune chunking parameters per content type.
    #      Defining a default here means a subclass that forgets to override metadata
    #      still produces valid (if non-optimal) chunks rather than crashing.
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    # What: Abstract method that every concrete handler must implement.
    # Why: Forces subclasses to define how their specific document type is split into
    #      chunks. The abstract declaration here means calling chunk() on the base class
    #      directly raises a TypeError, preventing accidental misuse.
    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


# What: Concrete handler that chunks social-media post content.
# Why: Posts are short, conversational content — smaller chunk_size (250) with
#      proportionally smaller overlap (25) produces tighter, more focused chunks
#      than the default, improving retrieval precision for short-form content.
class PostChunkingHandler(ChunkingDataHandler):

    # What: Override metadata with post-specific chunking parameters.
    # Why: Posts are shorter than articles or code; using 250-character chunks
    #      prevents a single post being split into only one trivially small chunk
    #      or over-split into fragments that lose coherent meaning.
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        # What: Initialise an empty list to accumulate the PostChunk objects.
        # Why: We process each text chunk individually in the loop below and collect
        #      results before returning them as a single list.
        data_models_list = []

        # What: Extract the cleaned text content from the input document.
        # Why: chunk_text operates on a plain string; extracting it here makes the
        #      function call below cleaner and signals intent clearly.
        cleaned_content = data_model.content

        # What: Split the post content into overlapping text chunks.
        # Why: chunk_text uses a sliding window of chunk_size characters with
        #      chunk_overlap characters of context carried over between adjacent chunks.
        #      Overlap preserves sentences that would otherwise be split at a boundary,
        #      keeping each chunk semantically coherent.
        chunks = chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            # What: Generate a deterministic ID by MD5-hashing the chunk text.
            # Why: The same text always produces the same hash, so if this post is
            #      re-processed, the chunks will have the same IDs and Qdrant's upsert
            #      will update rather than duplicate them.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            # What: Construct a PostChunk domain model for this text piece.
            # Why: PostChunk is the typed domain object for Qdrant — it carries the
            #      text, lineage (document_id, author_id), source metadata, and the
            #      optional image URL specific to posts. Pydantic validates all fields.
            model = PostChunk(
                # What: Wrap the MD5 hex string in UUID() to get a valid UUID4 object.
                # Why: The domain model and Qdrant both require a proper UUID, not a raw hex string.
                id=UUID(chunk_id, version=4),
                content=chunk,                          # The actual text of this chunk
                platform=data_model.platform,           # Preserve the source platform for attribution
                document_id=data_model.id,              # Link back to the parent cleaned document
                author_id=data_model.author_id,         # Link back to the author for personalisation
                author_full_name=data_model.author_full_name,  # Denormalised for display without extra lookups
                image=data_model.image if data_model.image else None,  # Carry optional image URL forward
                metadata=self.metadata,                 # Attach chunk_size/overlap for auditability
            )
            data_models_list.append(model)

        return data_models_list


# What: Concrete handler that chunks long-form article content (e.g. Medium posts).
# Why: Articles are much longer than posts and have natural paragraph/section structure.
#      chunk_article uses semantic splitting (min/max length) rather than raw character
#      count, producing chunks aligned with meaningful content boundaries.
class ArticleChunkingHandler(ChunkingDataHandler):

    # What: Override metadata with article-specific chunking parameters.
    # Why: min_length=1000 / max_length=2000 character bounds let chunk_article produce
    #      chunks large enough to be semantically rich but small enough to fit within
    #      the embedding model's context window. Raw character overlap is not used
    #      because chunk_article splits on sentence/paragraph boundaries instead.
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 2000,
        }

    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        # What: Initialise an empty list to collect ArticleChunk objects.
        # Why: Same accumulator pattern as PostChunkingHandler for consistency.
        data_models_list = []

        cleaned_content = data_model.content

        # What: Split the article using semantic-aware chunking.
        # Why: chunk_article respects sentence and paragraph boundaries within the
        #      min/max character range, producing coherent chunks without cutting
        #      a sentence in half mid-thought — important for dense article content.
        chunks = chunk_article(
            cleaned_content, min_length=self.metadata["min_length"], max_length=self.metadata["max_length"]
        )

        for chunk in chunks:
            # What: Deterministic chunk ID from MD5 hash of the text.
            # Why: Same idempotency reason as PostChunkingHandler — re-runs don't
            #      create duplicate Qdrant points for the same article text.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            # What: Construct an ArticleChunk with the article-specific `link` field.
            # Why: Articles have a canonical URL that posts don't. Storing it on the
            #      chunk means the RAG system can cite the exact source article in responses.
            model = ArticleChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,                   # Article-specific: source URL for citation
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


# What: Concrete handler that chunks code repository content (e.g. GitHub READMEs, code files).
# Why: Repository content is typically structured code or markdown — larger chunks
#      (1500 chars) with more overlap (100) preserve function signatures, class
#      definitions, and surrounding context that would be meaningless if split too finely.
class RepositoryChunkingHandler(ChunkingDataHandler):

    # What: Override metadata with repository-specific chunking parameters.
    # Why: Code snippets and README sections need larger windows (1500 chars) to
    #      stay meaningful. The larger overlap (100) ensures that a function signature
    #      on one chunk is repeated at the start of the next, maintaining context
    #      for the embedding model and for RAG retrieval.
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 1500,
            "chunk_overlap": 100,
        }

    def chunk(self, data_model: CleanedRepositoryDocument) -> list[RepositoryChunk]:
        # What: Initialise an empty list to collect RepositoryChunk objects.
        # Why: Same accumulator pattern used across all handlers for consistency.
        data_models_list = []

        cleaned_content = data_model.content

        # What: Split repository content using sliding-window character-based chunking.
        # Why: Unlike articles, repository text does not have consistent semantic
        #      boundaries (paragraphs), so character-count splitting with generous
        #      overlap is more reliable for code and structured markdown.
        chunks = chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            # What: Deterministic chunk ID from MD5 hash of the text.
            # Why: Idempotency — re-running the pipeline on the same repo produces
            #      the same chunk IDs, so Qdrant upserts rather than duplicates.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            # What: Construct a RepositoryChunk with both `name` and `link` fields.
            # Why: Repository chunks carry both the repo name (human-readable identifier)
            #      and the URL (direct link for citation), which are unique to this type
            #      and absent from post and article chunks.
            model = RepositoryChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                name=data_model.name,                   # Repository-specific: repo name (e.g. "decodingml/llm-twin")
                link=data_model.link,                   # Repository-specific: URL for source attribution
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list
