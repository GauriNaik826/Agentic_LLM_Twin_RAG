# What: Import ABC and abstractmethod to define the abstract base handler.
# Why: ABC prevents EmbeddingDataHandler from being instantiated directly;
#      abstractmethod forces every concrete subclass to implement map_model(),
#      guaranteeing that each handler knows how to convert its specific (chunk, embedding)
#      pair into the correct EmbeddedChunk subclass.
from abc import ABC, abstractmethod

# What: Import Generic, TypeVar, and cast from typing.
# Why: Generic[ChunkT, EmbeddedChunkT] makes each concrete handler's input/output
#      types explicit to static type checkers (e.g. ArticleChunk → EmbeddedArticleChunk).
#      cast() is used to tell the type checker that an embedding is list[float] without
#      a runtime conversion, since the embedding model returns an untyped list.
from typing import Generic, TypeVar, cast

# What: Import the singleton embedding model.
# Why: All handlers share the same embedding model instance. Using a singleton means
#      the heavy model weights are loaded once at module load time and reused for every
#      embed_batch() call, avoiding repeated loading overhead.
from llm_engineering.application.networks import EmbeddingModelSingleton

# What: Import the plain chunk domain models (input types for each handler).
# Why: Each concrete handler's map_model() accepts a specific chunk type; importing
#      them here provides precise type annotations for IDE support and static analysis.
from llm_engineering.domain.chunks import ArticleChunk, Chunk, PostChunk, RepositoryChunk

# What: Import the embedded chunk domain models (output types for each handler).
# Why: Each handler returns a specific EmbeddedChunk subclass that carries the vector
#      embedding alongside the chunk's fields; importing them here keeps the mapping
#      between chunk type and embedded chunk type explicit in this file.
from llm_engineering.domain.embedded_chunks import (
    EmbeddedArticleChunk,
    EmbeddedChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)

# What: Import the Query and EmbeddedQuery domain models used at inference time.
# Why: At RAG query time, a user's question is treated like a chunk — it needs to be
#      embedded with the same model so its vector is comparable to stored chunk vectors.
#      Query/EmbeddedQuery are the domain types for this special case.
from llm_engineering.domain.queries import EmbeddedQuery, Query

# What: TypeVar for the input chunk type, bound to Chunk.
# Why: Allows Generic[ChunkT, EmbeddedChunkT] to enforce that the input must be
#      a Chunk subclass (PostChunk, ArticleChunk, etc.), not an arbitrary type.
ChunkT = TypeVar("ChunkT", bound=Chunk)

# What: TypeVar for the output embedded chunk type, bound to EmbeddedChunk.
# Why: Pairs with ChunkT so the type checker knows each handler consistently maps
#      a specific chunk type to the corresponding embedded chunk type.
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

# What: Create the singleton embedding model instance at module load time.
# Why: Loading the embedding model (e.g. SentenceTransformer) is expensive — it
#      downloads weights and allocates GPU/CPU memory. Creating it once here as a
#      module-level variable ensures all four handler subclasses share the exact
#      same loaded model, saving memory and startup time.
embedding_model = EmbeddingModelSingleton()


# What: Abstract base class that all concrete embedding handlers must inherit from.
# Why: Defines the shared embedding logic (embed, embed_batch) once so subclasses
#      only need to implement map_model() — the one thing that differs between types.
#      This is the Template Method pattern: the algorithm skeleton lives in the base class,
#      and subclasses fill in the single type-specific step.
class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        """What: Embed a single chunk by delegating to embed_batch with a one-item list.
        Why: Provides a convenient single-item API without duplicating the batching logic.
        The [0] at the end unwraps the one-element list back to a single EmbeddedChunk.
        """
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        """What: Embed a batch of chunks in a single model forward pass and return
        a matching list of EmbeddedChunk objects.
        Why: Running the embedding model once on a batch is far more efficient than
        N individual forward passes — the model processes all inputs in parallel on GPU/CPU.
        This is the primary method called by EmbeddingDispatcher.dispatch() in the pipeline.
        """
        # What: Extract just the text content from each chunk into a plain list of strings.
        # Why: The embedding model only needs the raw text; passing full Pydantic objects
        #      would require the model to know about their structure. Building a plain list
        #      here decouples the model from the domain layer.
        embedding_model_input = [data_model.content for data_model in data_model]

        # What: Run the embedding model on all texts at once and get back a list of float lists.
        # Why: to_list=True converts the model's output (often a numpy array or tensor)
        #      into plain Python lists so the results are JSON-serialisable and Qdrant-compatible
        #      without any further conversion.
        embeddings = embedding_model(embedding_model_input, to_list=True)

        # What: Pair each chunk with its corresponding embedding and call map_model()
        #       to produce a typed EmbeddedChunk domain object.
        # Why: zip(data_model, embeddings) aligns each input chunk with its output vector
        #      by position. map_model() is the subclass-specific step that knows which
        #      EmbeddedChunk subclass to create (e.g. PostChunk → EmbeddedPostChunk).
        #      cast() tells the type checker that each embedding is list[float] without
        #      a costly runtime conversion.
        #      strict=False in zip means mismatched lengths are silently truncated — safe
        #      here because embedding_model always returns the same number of vectors as inputs.
        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    # What: Abstract method that each concrete handler must implement.
    # Why: This is the only step that differs between handlers — constructing the correct
    #      EmbeddedChunk subclass with the right type-specific fields (e.g. link for articles,
    #      image for posts). Declaring it abstract here forces subclasses to define it.
    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


# What: Concrete handler for embedding inference-time user queries.
# Why: At RAG query time, the user's question must be embedded with the same model
#      used for chunks so the vectors exist in the same semantic space and cosine
#      similarity comparisons are meaningful. This handler wraps query text in an
#      EmbeddedQuery rather than an EmbeddedChunk.
class QueryEmbeddingHandler(EmbeddingDataHandler):

    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        """What: Construct an EmbeddedQuery from a plain Query and its embedding vector.
        Why: EmbeddedQuery is the domain object used by the RAG retrieval step to
        perform the Qdrant vector search — it carries both the query text and the
        vector needed for approximate nearest-neighbour lookup.
        """
        return EmbeddedQuery(
            id=data_model.id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            content=data_model.content,
            # What: The dense float-list vector produced by the embedding model.
            # Why: This vector is what Qdrant compares against stored chunk vectors
            #      using cosine similarity to find the most relevant chunks.
            embedding=embedding,
            # What: Metadata dict recording which model and settings produced this embedding.
            # Why: Storing model provenance on the embedded object makes it possible to detect
            #      when stored embeddings are stale (produced by a different/older model version)
            #      and need to be regenerated.
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


# What: Concrete handler for embedding social-media post chunks.
# Why: PostChunk → EmbeddedPostChunk. The only difference from other handlers is
#      that EmbeddedPostChunk accepts no extra type-specific fields (posts have no link,
#      no repo name), so this map_model is the simplest of the content-type handlers.
class PostEmbeddingHandler(EmbeddingDataHandler):

    def map_model(self, data_model: PostChunk, embedding: list[float]) -> EmbeddedPostChunk:
        """What: Construct an EmbeddedPostChunk by copying all PostChunk fields and adding the embedding.
        Why: The id is preserved (same MD5-based UUID from chunking) so Qdrant's upsert
        on re-runs updates the existing point rather than creating a duplicate.
        All lineage fields (document_id, author_id) are carried forward so the embedded
        chunk remains traceable back to the original raw document.
        """
        return EmbeddedPostChunk(
            id=data_model.id,                           # Preserve deterministic chunk ID for idempotent upserts
            content=data_model.content,                 # The text that was embedded — kept for retrieval display
            embedding=embedding,                        # The dense vector for ANN search in Qdrant
            platform=data_model.platform,               # Source platform preserved for attribution in RAG responses
            document_id=data_model.document_id,         # Lineage: which cleaned document this chunk came from
            author_id=data_model.author_id,             # Lineage: which author owns this content
            author_full_name=data_model.author_full_name,  # Denormalised for display without extra DB lookups
            # What: Embed model provenance metadata.
            # Why: Records exactly which model version produced this vector so staleness
            #      can be detected if the embedding model is upgraded in the future.
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


# What: Concrete handler for embedding article chunks.
# Why: ArticleChunk → EmbeddedArticleChunk. Articles carry a `link` field that posts
#      and repos don't share; this handler passes it through so the source URL is
#      available in every embedded article chunk for citation in RAG responses.
class ArticleEmbeddingHandler(EmbeddingDataHandler):

    def map_model(self, data_model: ArticleChunk, embedding: list[float]) -> EmbeddedArticleChunk:
        """What: Construct an EmbeddedArticleChunk from an ArticleChunk and its embedding.
        Why: EmbeddedArticleChunk adds `link` on top of the base EmbeddedChunk fields,
        enabling the RAG system to return a direct source URL alongside retrieved content.
        """
        return EmbeddedArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            # What: Article-specific canonical URL carried forward from the chunk.
            # Why: Preserved all the way to the embedded chunk so the RAG response
            #      can cite exactly which article the retrieved passage came from.
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


# What: Concrete handler for embedding repository chunks.
# Why: RepositoryChunk → EmbeddedRepositoryChunk. Repositories carry both `name`
#      (human-readable repo identifier) and `link` (URL); both are preserved on the
#      embedded chunk so code-sourced RAG responses can cite the exact repo.
class RepositoryEmbeddingHandler(EmbeddingDataHandler):

    def map_model(self, data_model: RepositoryChunk, embedding: list[float]) -> EmbeddedRepositoryChunk:
        """What: Construct an EmbeddedRepositoryChunk from a RepositoryChunk and its embedding.
        Why: EmbeddedRepositoryChunk adds `name` and `link` on top of EmbeddedChunk,
        so the RAG system can attribute retrieved code snippets to the specific repository
        and provide a direct link for the user to follow.
        """
        return EmbeddedRepositoryChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            # What: Repository name carried forward from the chunk.
            # Why: Human-readable identifier displayed in RAG responses so users know
            #      which repository the retrieved code/README text came from.
            name=data_model.name,
            # What: Repository URL carried forward from the chunk.
            # Why: Provides a direct link back to the source repo, enabling users to
            #      inspect the full context of a retrieved code snippet.
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
