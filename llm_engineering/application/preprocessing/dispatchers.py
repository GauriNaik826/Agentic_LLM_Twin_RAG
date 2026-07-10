# What: Import loguru logger for structured, levelled runtime logging.
# Why: Provides clear audit logs after each dispatch (document cleaned/chunked/embedded)
#      so operators can monitor pipeline progress without opening the full artifact.
from loguru import logger

# What: Import the two base document classes used as type annotations in dispatcher signatures.
# Why: NoSQLBaseDocument covers raw MongoDB documents (input to cleaning);
#      VectorBaseDocument covers Qdrant-backed documents (cleaned docs, chunks, embedded chunks).
#      Using base classes in signatures keeps dispatchers generic — they work with any subtype.
from llm_engineering.domain.base import NoSQLBaseDocument, VectorBaseDocument

# What: Import DataCategory, the StrEnum of content categories (POSTS, ARTICLES, REPOSITORIES, QUERIES).
# Why: Every factory uses DataCategory as its routing key to decide which handler to instantiate.
#      Using an enum prevents typos and makes the branching logic readable.
from llm_engineering.domain.types import DataCategory

# What: Import all three concrete chunking handlers and the abstract base.
# Why: ChunkingHandlerFactory needs to instantiate the correct handler per category;
#      ChunkingDataHandler is used as the return type annotation so callers know the
#      factory always returns something with a .chunk() method.
from .chunking_data_handlers import (
    ArticleChunkingHandler,
    ChunkingDataHandler,
    PostChunkingHandler,
    RepositoryChunkingHandler,
)

# What: Import all three concrete cleaning handlers and the abstract base.
# Why: Same pattern as chunking — CleaningHandlerFactory picks the right handler;
#      CleaningDataHandler is the return type so callers can call .clean() on the result.
from .cleaning_data_handlers import (
    ArticleCleaningHandler,
    CleaningDataHandler,
    PostCleaningHandler,
    RepositoryCleaningHandler,
)

# What: Import all concrete embedding handlers, the abstract base, and the special query handler.
# Why: EmbeddingHandlerFactory needs all four options; QueryEmbeddingHandler handles
#      inference-time query embedding separately from document embedding during training.
from .embedding_data_handlers import (
    ArticleEmbeddingHandler,
    EmbeddingDataHandler,
    PostEmbeddingHandler,
    QueryEmbeddingHandler,
    RepositoryEmbeddingHandler,
)


# ---------------------------------------------------------------------------
# CLEANING — Factory + Dispatcher
# ---------------------------------------------------------------------------

# What: Factory class responsible for creating the correct CleaningDataHandler
#       based on the document's DataCategory.
# Why: Centralising handler creation in a factory means the dispatcher never
#      contains if/elif logic — it just asks the factory for the right handler.
#      This is the Factory Method pattern: new document types only require adding
#      a branch here, not changing any other class.
class CleaningHandlerFactory:

    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        """What: Instantiate and return the cleaning handler matching the category.
        Why: Each content type (post, article, repository) needs different cleaning
        logic (e.g. HTML stripping differs between Medium articles and GitHub README).
        The factory maps category → handler so the dispatcher stays type-agnostic.
        Raises ValueError for unknown categories to catch misconfiguration early.
        """
        if data_category == DataCategory.POSTS:
            return PostCleaningHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleCleaningHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryCleaningHandler()
        else:
            # What: Raise for any unrecognised category.
            # Why: Failing loudly here is safer than silently returning a no-op handler
            #      that would let uncleaned data pass through to the vector DB.
            raise ValueError("Unsupported data type")


# What: Dispatcher that orchestrates the entire cleaning step for a single raw document.
# Why: The dispatcher is the "glue" between the pipeline step (clean_documents)
#      and the type-specific handler. It reads the document's category, delegates
#      to the factory, runs the handler, and logs the result — all in one place.
#      This is the Strategy pattern: the algorithm (cleaning) is selected at runtime.
class CleaningDispatcher:

    # What: Class-level factory instance shared by all calls to dispatch().
    # Why: Creating one factory at class definition time avoids re-instantiating it
    #      on every dispatch() call; the factory itself is stateless so sharing is safe.
    factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        """What: Determine the document's category, select the correct cleaning handler,
        run it, log the result, and return the cleaned document.
        Why: Having a single dispatch() entry point means the pipeline step (clean_documents)
        only ever calls CleaningDispatcher.dispatch(doc) — it is completely decoupled from
        the details of how each type is cleaned.
        """
        # What: Read the document's MongoDB collection name and convert it to a DataCategory enum.
        # Why: get_collection_name() returns the Settings.name string (e.g. "posts");
        #      DataCategory() wraps it as an enum value so the factory's if/elif branches
        #      can compare with enum constants rather than raw strings.
        data_category = DataCategory(data_model.get_collection_name())

        # What: Ask the factory to create the right handler for this category.
        # Why: Delegating to the factory keeps dispatch() free of type-switching logic;
        #      the factory owns all knowledge of which handler maps to which category.
        handler = cls.factory.create_handler(data_category)

        # What: Run the handler's clean() method to produce a CleanedDocument.
        # Why: The handler contains the actual cleaning logic (HTML stripping,
        #      whitespace normalisation, etc.) specific to this document type.
        clean_model = handler.clean(data_model)

        # What: Log a structured info message with category and cleaned content length.
        # Why: Provides per-document visibility in the logs so operators can spot
        #      documents that cleaned down to suspiciously short content.
        logger.info(
            "Document cleaned successfully.",
            data_category=data_category,
            cleaned_content_len=len(clean_model.content),
        )

        # What: Return the cleaned document (a VectorBaseDocument subclass).
        # Why: The pipeline step collects these into a list and returns them as the
        #      "cleaned_documents" ZenML artifact for the next step.
        return clean_model


# ---------------------------------------------------------------------------
# CHUNKING — Factory + Dispatcher
# ---------------------------------------------------------------------------

# What: Factory class responsible for creating the correct ChunkingDataHandler
#       based on the cleaned document's DataCategory.
# Why: Same Factory Method pattern as CleaningHandlerFactory — isolates handler
#      selection so ChunkingDispatcher stays clean. Adding a new content type
#      only requires a new branch here.
class ChunkingHandlerFactory:

    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkingDataHandler:
        """What: Instantiate and return the chunking handler matching the category.
        Why: Each content type requires different chunking parameters and strategies
        (posts use small sliding-window chunks; articles use semantic splitting;
        repositories use large windows with more overlap). The factory encapsulates
        that mapping so nothing else in the codebase needs to know about it.
        """
        if data_category == DataCategory.POSTS:
            return PostChunkingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleChunkingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


# What: Dispatcher that orchestrates the chunking step for a single cleaned document.
# Why: Same role as CleaningDispatcher but for the chunking stage. Keeps the pipeline
#      step (rag.py chunk_and_embed) decoupled from handler selection.
class ChunkingDispatcher:

    # What: Reference to the factory class (not an instance) as a class attribute.
    # Why: create_handler is a @staticmethod so we don't need an instance;
    #      storing the class lets us call cls.factory.create_handler() without
    #      the overhead of constructing a factory object on every dispatch call.
    factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument) -> list[VectorBaseDocument]:
        """What: Select the right chunking handler, split the document into chunks,
        log the result, and return the list of chunk domain objects.
        Why: Single entry point for chunking keeps the pipeline step simple:
        it just calls ChunkingDispatcher.dispatch(doc) for each document.
        """
        # What: Read the DataCategory directly from the VectorBaseDocument's Config.
        # Why: Cleaned documents already live in Qdrant and carry a Config.category;
        #      no string-to-enum conversion needed here (unlike CleaningDispatcher
        #      which reads from a MongoDB collection name string).
        data_category = data_model.get_category()

        # What: Ask the factory for the correct chunking handler.
        # Why: Keeps dispatch() free of if/elif type-switching logic.
        handler = cls.factory.create_handler(data_category)

        # What: Run chunk() to split the cleaned document into a list of Chunk objects.
        # Why: Each handler applies its own strategy (size, overlap, semantic splitting)
        #      and returns a typed list (e.g. list[ArticleChunk]).
        chunk_models = handler.chunk(data_model)

        # What: Log how many chunks were produced and for which category.
        # Why: Lets operators quickly verify chunking output volume per document type
        #      without loading the full artifact.
        logger.info(
            "Document chunked successfully.",
            num=len(chunk_models),
            data_category=data_category,
        )

        return chunk_models


# ---------------------------------------------------------------------------
# EMBEDDING — Factory + Dispatcher
# ---------------------------------------------------------------------------

# What: Factory class responsible for creating the correct EmbeddingDataHandler
#       based on the chunk's DataCategory (or QUERIES for inference-time embedding).
# Why: Same Factory Method pattern. The extra QUERIES branch is what separates
#      this factory from the others — query embedding at inference time uses the
#      same embedding model but a different handler tailored for short query text.
class EmbeddingHandlerFactory:

    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        """What: Instantiate and return the embedding handler matching the category.
        Why: All chunk types use the same underlying embedding model, but the handler
        may wrap the result in different EmbeddedChunk subclasses (EmbeddedPostChunk,
        EmbeddedArticleChunk, etc.) so the correct Qdrant collection is targeted.
        QUERIES gets its own handler because at inference time we embed a user query
        string, not a domain document — the handler signature is slightly different.
        """
        if data_category == DataCategory.QUERIES:
            return QueryEmbeddingHandler()
        if data_category == DataCategory.POSTS:
            return PostEmbeddingHandler()
        elif data_category == DataCategory.ARTICLES:
            return ArticleEmbeddingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


# What: Dispatcher that orchestrates the embedding step for one document or a batch.
# Why: Embedding is typically done in batches (see rag.py where chunks are grouped
#      into batches of 10 before calling EmbeddingDispatcher.dispatch). This dispatcher
#      handles both a single document and a list, normalising the input before
#      delegating to the handler's embed_batch() method.
class EmbeddingDispatcher:

    # What: Reference to the factory class (not an instance).
    # Why: Same rationale as ChunkingDispatcher — create_handler is static,
    #      so no instance is needed; avoiding instantiation reduces overhead.
    factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch(
        cls, data_model: VectorBaseDocument | list[VectorBaseDocument]
    ) -> VectorBaseDocument | list[VectorBaseDocument]:
        """What: Accept either a single document or a list, embed them in batch,
        and return the same shape (single → single, list → list).
        Why: Callers in the pipeline always pass batches (for efficiency), but
        inference-time callers may pass a single query. Normalising to a list
        internally means embed_batch() only needs one implementation path.
        """
        # What: Record whether the input was a single document or a list.
        # Why: We need to unwrap the result back to a single item at the end
        #      if the caller passed a single document, maintaining a consistent API.
        is_list = isinstance(data_model, list)

        # What: Wrap a single document in a list so the rest of the method is uniform.
        # Why: embed_batch() always operates on a list; this normalisation avoids
        #      duplicating the batch logic for the single-item case.
        if not is_list:
            data_model = [data_model]

        # What: Return an empty list immediately if there are no documents to embed.
        # Why: Calling embed_batch() with an empty list would waste a model forward
        #      pass / API call and could cause index-out-of-bounds errors in the handler.
        if len(data_model) == 0:
            return []

        # What: Read the DataCategory from the first document in the batch.
        # Why: All items in a batch must share the same category (enforced below),
        #      so the first item's category is representative of the whole batch.
        data_category = data_model[0].get_category()

        # What: Assert that every document in the batch shares the same category.
        # Why: Each category maps to a different EmbeddedChunk type and Qdrant collection.
        #      A mixed batch would send some chunks to the wrong collection, corrupting the
        #      vector store. Failing loudly here prevents silent data misrouting.
        assert all(
            data_model.get_category() == data_category for data_model in data_model
        ), "Data models must be of the same category."

        # What: Ask the factory for the correct embedding handler for this category.
        # Why: The handler wraps the raw embedding vector in the correct EmbeddedChunk
        #      subclass (PostEmbeddingHandler → EmbeddedPostChunk, etc.) so the result
        #      can be routed to the right Qdrant collection without any further branching.
        handler = cls.factory.create_handler(data_category)

        # What: Run embed_batch() to compute embeddings for all documents in the batch.
        # Why: Batching multiple documents in a single model forward pass is significantly
        #      more efficient than embedding one document at a time.
        embedded_chunk_model = handler.embed_batch(data_model)

        # What: Unwrap the list back to a single item if the caller passed a single document.
        # Why: Preserves the caller's expected return type — if they passed one document,
        #      they get one embedded chunk back, not a one-element list.
        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        # What: Log a structured info message confirming successful embedding.
        # Why: Provides an audit trail showing which category was just embedded,
        #      useful for debugging embedding failures mid-pipeline.
        logger.info(
            "Data embedded successfully.",
            data_category=data_category,
        )

        return embedded_chunk_model
