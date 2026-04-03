# What: Import Python's built-in uuid module for UUID generation and manipulation.
# Why: Used to generate unique document IDs (uuid.uuid4) and to convert raw
#      string IDs returned by Qdrant back into proper UUID objects (UUID(..., version=4)).
import uuid

# What: Import ABC (Abstract Base Class) marker.
# Why: Marks VectorBaseDocument as abstract so it can never be instantiated directly —
#      only concrete subclasses (EmbeddedPostChunk, etc.) should be created.
from abc import ABC

# What: Import type-hinting utilities: Any, Callable, Dict, Generic, Type, TypeVar.
# Why: Generic[T] makes classmethods return the correct concrete subclass type;
#      Callable is needed to type the `selector` argument in _group_by;
#      Dict, Type, Any are used for precise return-type annotations throughout.
from typing import Any, Callable, Dict, Generic, Type, TypeVar

# What: Import UUID for type checking and conversion of Qdrant's string-based IDs.
# Why: Qdrant stores point IDs as strings; we convert them back to UUID objects
#      so the rest of the codebase can work with properly typed identifiers.
from uuid import UUID

# What: Import numpy for array type checking during vector serialisation.
# Why: Embedding models (e.g. SentenceTransformers) return numpy arrays.
#      PyMongo/Qdrant cannot serialise np.ndarray directly, so we detect and
#      convert them to plain Python lists before sending to Qdrant.
import numpy as np

# What: Import loguru's logger for structured, levelled logging.
# Why: Provides clear runtime visibility (info on collection creation, error on
#      insert failure) without configuring Python's standard logging module.
from loguru import logger

# What: Import Pydantic types: UUID4 (validated UUID field), BaseModel (base class),
#       and Field (field configuration helper).
# Why: BaseModel provides automatic field validation and serialisation;
#      UUID4 ensures ID fields are valid v4 UUIDs; Field(default_factory=uuid.uuid4)
#      generates a fresh UUID per instance avoiding shared-mutable-default issues.
from pydantic import UUID4, BaseModel, Field

# What: Import Qdrant's HTTP exceptions module.
# Why: UnexpectedResponse is raised when a collection doesn't exist yet.
#      Catching it specifically allows auto-creation of the collection on first insert
#      rather than crashing the pipeline.
from qdrant_client.http import exceptions

# What: Import Qdrant's Distance enum and VectorParams config model.
# Why: Distance.COSINE specifies the similarity metric for the vector index;
#      VectorParams bundles vector size + distance metric when creating a new collection.
from qdrant_client.http.models import Distance, VectorParams

# What: Import Qdrant models: CollectionInfo (collection metadata), PointStruct
#       (a single Qdrant point with id + vector + payload), Record (a point returned by scroll).
# Why: These are the native Qdrant types used to read from and write to Qdrant's API.
from qdrant_client.models import CollectionInfo, PointStruct, Record

# What: Import the singleton embedding model used to determine the embedding vector size.
# Why: When creating a collection with a vector index, Qdrant requires the vector
#      dimension upfront. EmbeddingModelSingleton().embedding_size provides that value
#      without instantiating the model multiple times.
from llm_engineering.application.networks.embeddings import EmbeddingModelSingleton

# What: Import the domain-level configuration error.
# Why: Raised when a subclass forgets to declare Config.name, Config.category, or
#      Config.use_vector_index, giving a clear message instead of a generic AttributeError.
from llm_engineering.domain.exceptions import ImproperlyConfigured

# What: Import the DataCategory StrEnum that centralises collection/category name constants.
# Why: Using an enum instead of raw strings prevents typos and makes category names
#      refactor-safe across the entire codebase.
from llm_engineering.domain.types import DataCategory

# What: Import the Qdrant client connection singleton.
# Why: All Qdrant operations (upsert, scroll, search, create_collection) go through
#      this shared client so the application maintains one connection pool.
from llm_engineering.infrastructure.db.qdrant import connection

# What: TypeVar bound to VectorBaseDocument so Generic classmethods always return
#       the correct concrete subclass type (e.g. EmbeddedArticleChunk, not the base).
# Why: Without this, ArticleChunk.find(...) would be typed as VectorBaseDocument,
#      losing all subclass-specific field information for static analysis.
T = TypeVar("T", bound="VectorBaseDocument")


# What: Abstract base class providing Qdrant persistence for all vector-store domain models.
# Why: Cleaned documents, chunks, and embedded chunks all need the same Qdrant operations
#      (insert, search, scroll, collection management). Centralising them here means each
#      concrete model only declares its own fields; everything else is inherited for free.
#      Inheriting from Generic[T] makes all classmethods properly typed for subclasses.
class VectorBaseDocument(BaseModel, Generic[T], ABC):

    # What: Auto-generated UUID4 primary key for every document.
    # Why: Qdrant requires each point to have a unique ID. Using default_factory=uuid.uuid4
    #      generates a fresh UUID per instance, avoiding the shared-mutable-default bug.
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        """What: Compare two documents for equality.
        Why: Two documents are the same if and only if they are the same type AND share
        the same id. The isinstance check prevents a PostChunk and ArticleChunk from
        being considered equal even if they somehow share a UUID (astronomically unlikely
        but defensively handled).
        """
        if not isinstance(value, self.__class__):
            return False

        # What: Compare by id only — the unique, stable identifier.
        # Why: Field values may change (content updated), but the id never changes,
        #      so id equality is the correct definition of "same document".
        return self.id == value.id

    def __hash__(self) -> int:
        """What: Make documents hashable so they can be stored in sets or used as dict keys.
        Why: Pydantic models are not hashable by default (they are mutable). We override
        __hash__ based on id alone, consistent with __eq__, so documents can be
        deduplicated with set() in the pipeline metadata helpers.
        """
        return hash(self.id)

    @classmethod
    def from_record(cls: Type[T], point: Record) -> T:
        """What: Deserialise a raw Qdrant Record (point) into a typed domain object.
        Why: Qdrant returns points with a string id and a flat payload dict. This method
        bridges that to a Pydantic model by converting the id back to a UUID and
        unpacking payload fields into the constructor.
        """
        # What: Convert Qdrant's string point ID back into a proper UUID object.
        # Why: The rest of the codebase uses UUID4; Qdrant stores IDs as strings.
        #      Explicitly passing version=4 validates that the string is a v4 UUID.
        _id = UUID(point.id, version=4)

        # What: Extract the payload dict, defaulting to empty if absent.
        # Why: Qdrant may return None for the payload if with_payload=False was used;
        #      the `or {}` guard prevents a TypeError when unpacking below.
        payload = point.payload or {}

        # What: Merge the id with the rest of the payload fields into one dict.
        # Why: The Pydantic constructor expects a single flat dict of all fields;
        #      this combines the separately stored id with the payload fields.
        attributes = {
            "id": _id,
            **payload,
        }

        # What: If the concrete subclass has an `embedding` field, attach the point vector.
        # Why: CleanedDocuments and Chunks use VectorBaseDocument but have no embedding field;
        #      EmbeddedChunks do. _has_class_attribute checks the class hierarchy so we only
        #      set `embedding` when the subclass actually declares it.
        if cls._has_class_attribute("embedding"):
            attributes["embedding"] = point.vector or None

        # What: Construct and return a fully validated Pydantic instance.
        # Why: Passing **attributes through the constructor ensures all types are
        #      coerced and validated (e.g. string UUIDs → UUID4 objects).
        return cls(**attributes)

    def to_point(self: T, **kwargs) -> PointStruct:
        """What: Serialise this domain object into a Qdrant PointStruct ready for insertion.
        Why: Qdrant expects a PointStruct with a string id, a vector list, and a flat payload
        dict. This method converts the Pydantic model into exactly that shape.
        """
        # What: Pop serialisation options from kwargs, falling back to safe defaults.
        # Why: Allows callers to customise serialisation (e.g. exclude_unset=True)
        #      without changing the method signature. pop() removes them so they don't
        #      leak into the model_dump call below.
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        # What: Serialise the Pydantic model to a plain Python dict (with UUIDs as strings).
        # Why: model_dump provides a consistent, validated view of all fields.
        #      Our overridden model_dump also stringifies UUIDs so Qdrant can serialise them.
        payload = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        # What: Remove "id" from the payload and convert it to a plain string.
        # Why: Qdrant's PointStruct takes id as a separate top-level field, not inside
        #      the payload. pop() removes it from the dict so it isn't stored twice.
        _id = str(payload.pop("id"))

        # What: Remove the "embedding" field from the payload to use as the vector.
        # Why: PointStruct separates the vector field from metadata payload.
        #      Documents without an embedding field return {} from .pop(), which is fine —
        #      Qdrant accepts an empty vector for collections without a vector index.
        vector = payload.pop("embedding", {})

        # What: Convert numpy arrays to Python lists if the embedding is an ndarray.
        # Why: Qdrant's client cannot serialise numpy arrays; it requires a plain list of floats.
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # What: Construct and return the Qdrant PointStruct.
        # Why: PointStruct is the native Qdrant type accepted by upsert/insert_many;
        #      separating id, vector, and payload matches Qdrant's data model exactly.
        return PointStruct(id=_id, vector=vector, payload=payload)

    def model_dump(self: T, **kwargs) -> dict:
        """What: Override Pydantic's model_dump to stringify all UUID fields recursively.
        Why: Pydantic v2 returns UUID objects as-is. Qdrant's client cannot serialise
        uuid.UUID objects natively, so we convert them here once rather than
        repeating the conversion in every method that serialises a document.
        """
        # What: Call Pydantic's default model_dump to get the base dict.
        # Why: This handles all standard Pydantic serialisation (aliases, exclusions, etc.)
        #      before we apply our UUID-stringification post-processing.
        dict_ = super().model_dump(**kwargs)

        # What: Recursively walk the dict and convert any UUID values to strings.
        # Why: UUIDs may appear at any nesting level (top-level fields, nested dicts,
        #      lists of dicts). _uuid_to_str handles all cases in one pass.
        dict_ = self._uuid_to_str(dict_)

        return dict_

    def _uuid_to_str(self, item: Any) -> Any:
        """What: Recursively convert all UUID objects within a dict/list structure to strings.
        Why: Qdrant (and JSON serialisation in general) cannot handle uuid.UUID objects.
        This helper walks nested dicts and lists so no UUID escapes stringification,
        regardless of how deeply nested it is.
        """
        # What: Only process dict items; non-dict types (str, int, etc.) are returned as-is.
        # Why: The recursion base case — primitive values need no conversion.
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, UUID):
                    # What: Convert UUID objects directly to their canonical string form.
                    # Why: UUID.__str__ returns the standard hyphenated format (xxxxxxxx-xxxx-…)
                    #      which is the format Qdrant expects for point IDs and payload fields.
                    item[key] = str(value)
                elif isinstance(value, list):
                    # What: Recurse into lists to handle lists of dicts or UUIDs.
                    # Why: Fields like author IDs stored in a list would be missed without
                    #      this branch.
                    item[key] = [self._uuid_to_str(v) for v in value]
                elif isinstance(value, dict):
                    # What: Recurse into nested dicts (e.g. metadata sub-dicts).
                    # Why: The metadata field can contain arbitrary nested structures;
                    #      this ensures UUIDs inside metadata are also stringified.
                    item[key] = {k: self._uuid_to_str(v) for k, v in value.items()}

        return item

    @classmethod
    def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool:
        """What: Public entry point for inserting a list of documents into Qdrant.
        Why: Wraps _bulk_insert with auto-collection-creation logic. If the collection
        doesn't exist yet, it is created on the fly and the insert is retried once.
        This allows the pipeline to run without pre-provisioning Qdrant collections.
        """
        try:
            # What: Attempt the bulk insert into the (assumed) existing collection.
            # Why: On the happy path (collection already exists), this is the only call needed.
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            # What: Qdrant raises UnexpectedResponse when the collection does not exist.
            # Why: We treat this as a first-run signal and auto-create the collection
            #      rather than requiring manual setup before each pipeline run.
            logger.info(
                f"Collection '{cls.get_collection_name()}' does not exist. Trying to create the collection and reinsert the documents."
            )

            # What: Create the Qdrant collection with the correct vector config.
            # Why: create_collection reads Config.use_vector_index to decide whether
            #      to set up an HNSW index — only needed for embedded chunks.
            cls.create_collection()

            try:
                # What: Retry the insert after collection creation.
                # Why: The first attempt failed only because the collection was missing;
                #      now that it exists, the insert should succeed.
                cls._bulk_insert(documents)
            except exceptions.UnexpectedResponse:
                # What: If the retry also fails, log and return False.
                # Why: A second failure indicates a real problem (auth, schema, network)
                #      rather than a missing collection, so we surface it to the caller.
                logger.error(f"Failed to insert documents in '{cls.get_collection_name()}'.")

                return False

        # What: Return True to signal all documents were inserted successfully.
        # Why: The calling step (load_to_vector_db) checks this flag to decide whether
        #      to continue or abort the pipeline run.
        return True

    @classmethod
    def _bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> None:
        """What: Convert documents to Qdrant PointStructs and upsert them in one batch.
        Why: Separated from bulk_insert so the retry logic in the public method is clean.
        upsert (insert-or-update) is used so re-running the pipeline with the same
        documents is idempotent — existing points are updated rather than duplicated.
        """
        # What: Convert each domain document to a Qdrant PointStruct.
        # Why: Qdrant's upsert API only accepts PointStruct objects, not Pydantic models.
        points = [doc.to_point() for doc in documents]

        # What: Send all points to Qdrant in a single network call.
        # Why: upsert batches the operation server-side, which is much faster than
        #      N individual inserts and is idempotent on duplicate IDs.
        connection.upsert(collection_name=cls.get_collection_name(), points=points)

    @classmethod
    def bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        """What: Public entry point for paginated retrieval of all documents in a collection.
        Why: Wraps _bulk_find with error handling so callers always receive a tuple
        (list, next_offset) and never need to handle Qdrant exceptions themselves.
        Returns an empty list with None offset on failure so callers can iterate safely.
        """
        try:
            documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            # What: Log the failure and return a safe empty result.
            # Why: Returning ([], None) keeps callers simple — they don't need to
            #      distinguish between "collection empty" and "query failed".
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")

            documents, next_offset = [], None

        return documents, next_offset

    @classmethod
    def _bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        """What: Paginate through a Qdrant collection using Qdrant's scroll API.
        Why: Qdrant's scroll (cursor-based pagination) is the correct way to retrieve
        large numbers of points without loading everything into memory at once.
        Returns a next_offset UUID that callers pass back to retrieve the next page.
        """
        # What: Get the Qdrant collection name from the subclass Config.
        # Why: All Qdrant API calls need the collection name as a string argument.
        collection_name = cls.get_collection_name()

        # What: Extract the pagination offset from kwargs and convert it to a string.
        # Why: Qdrant's scroll API expects a string offset (the ID of the last seen point).
        #      We accept a UUID from callers and convert it here so the API contract is clean.
        offset = kwargs.pop("offset", None)
        offset = str(offset) if offset else None

        # What: Call Qdrant's scroll endpoint to retrieve a page of records.
        # Why: scroll returns (records, next_offset) where next_offset is None on the last page,
        #      making it easy to implement "load more" or full-collection iteration.
        #      with_payload=True includes the point's metadata; with_vectors=False skips the
        #      raw vector bytes to keep the response payload small.
        records, next_offset = connection.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            offset=offset,
            **kwargs,
        )

        # What: Deserialise each raw Qdrant Record into a typed domain model.
        # Why: The rest of the codebase works with Pydantic models, not raw Qdrant Records.
        documents = [cls.from_record(record) for record in records]

        # What: Convert Qdrant's string next_offset back to a UUID (or keep None for last page).
        # Why: The public API exposes UUID offsets for consistency with the rest of the
        #      codebase; callers pass the UUID back to the next bulk_find call.
        if next_offset is not None:
            next_offset = UUID(next_offset, version=4)

        return documents, next_offset

    @classmethod
    def search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        """What: Public entry point for approximate nearest-neighbour (ANN) vector search.
        Why: Wraps _search with error handling so callers always receive a list of typed
        documents and never need to handle Qdrant exceptions. Returns [] on failure.
        This is the primary RAG retrieval method used at inference time.
        """
        try:
            documents = cls._search(query_vector=query_vector, limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            # What: Log and return empty list on failure.
            # Why: A failed search should degrade gracefully (no results) rather than
            #      crashing the inference pipeline mid-request.
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")

            documents = []

        return documents

    @classmethod
    def _search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        """What: Execute an ANN vector search against the Qdrant collection.
        Why: Qdrant's search finds the `limit` most similar points to query_vector using
        the HNSW index. This is the core retrieval operation of the RAG pipeline —
        given an embedded user query, return the most semantically similar chunks.
        """
        collection_name = cls.get_collection_name()

        # What: Call Qdrant's search API with the query vector and options.
        # Why: with_payload=True returns the full metadata so we can reconstruct
        #      Pydantic models; with_vectors=False omits raw vectors to save bandwidth.
        records = connection.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            **kwargs,
        )

        # What: Deserialise each result into a typed domain model.
        # Why: Ensures callers receive fully validated Pydantic objects with all fields
        #      correctly typed, regardless of what Qdrant returned.
        documents = [cls.from_record(record) for record in records]

        return documents

    @classmethod
    def get_or_create_collection(cls: Type[T]) -> CollectionInfo:
        """What: Return the Qdrant collection metadata, creating the collection if it doesn't exist.
        Why: Idempotent collection setup — the pipeline can call this on every run without
        failing if the collection was already created in a previous run.
        Raises RuntimeError if collection creation itself fails, since that is unrecoverable.
        """
        collection_name = cls.get_collection_name()

        try:
            # What: Try to fetch the existing collection metadata.
            # Why: On all runs after the first, the collection already exists and this
            #      is the only call needed — no creation overhead.
            return connection.get_collection(collection_name=collection_name)
        except exceptions.UnexpectedResponse:
            # What: Collection does not exist yet — create it.
            # Why: First-run scenario; we read use_vector_index from Config to decide
            #      whether to configure an HNSW index (only for embedded chunk collections).
            use_vector_index = cls.get_use_vector_index()

            collection_created = cls._create_collection(
                collection_name=collection_name, use_vector_index=use_vector_index
            )

            # What: Raise if collection creation returned False (Qdrant-level failure).
            # Why: Without the collection the pipeline cannot proceed; raising here gives
            #      a clear error message rather than a confusing downstream failure.
            if collection_created is False:
                raise RuntimeError(f"Couldn't create collection {collection_name}") from None

            # What: Fetch and return the newly created collection's metadata.
            # Why: The caller may need the CollectionInfo (vector size, point count, etc.)
            #      to validate the collection was set up correctly.
            return connection.get_collection(collection_name=collection_name)

    @classmethod
    def create_collection(cls: Type[T]) -> bool:
        """What: Public method to explicitly create the Qdrant collection for this class.
        Why: Called by bulk_insert's auto-creation logic. Delegates to _create_collection
        after reading the collection name and vector index setting from Config.
        """
        collection_name = cls.get_collection_name()
        use_vector_index = cls.get_use_vector_index()

        return cls._create_collection(collection_name=collection_name, use_vector_index=use_vector_index)

    @classmethod
    def _create_collection(cls, collection_name: str, use_vector_index: bool = True) -> bool:
        """What: Create a Qdrant collection with the appropriate vector configuration.
        Why: Whether to create an HNSW vector index depends on the document stage:
          - Cleaned documents and plain chunks: no index needed (use_vector_index=False).
          - Embedded chunks: index required for fast ANN search (use_vector_index=True).
        The vector size is read from EmbeddingModelSingleton to stay in sync with
        whatever embedding model is currently configured.
        """
        if use_vector_index is True:
            # What: Configure HNSW indexing with cosine similarity for the embedding dimension.
            # Why: Cosine similarity is the standard metric for semantic text embeddings;
            #      the size must match the model's output dimension exactly or Qdrant will reject inserts.
            vectors_config = VectorParams(size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE)
        else:
            # What: Empty dict means "no vector index" — store points as payload-only.
            # Why: Cleaned docs and raw chunks are stored for lineage/retrieval by metadata,
            #      not by vector similarity. Skipping the index saves memory and index-build time.
            vectors_config = {}

        # What: Create the collection in Qdrant and return True/False indicating success.
        # Why: The return value bubbles up through create_collection → bulk_insert so the
        #      pipeline can detect and log collection-creation failures.
        return connection.create_collection(collection_name=collection_name, vectors_config=vectors_config)

    @classmethod
    def get_category(cls: Type[T]) -> DataCategory:
        """What: Return the DataCategory enum value declared on this document class.
        Why: Used by metadata helpers in the ZenML pipeline steps to group documents
        by category (posts, articles, repositories) for per-category statistics.
        Raises ImproperlyConfigured with a clear message if Config.category is missing,
        catching misconfigured subclasses at runtime rather than silently returning None.
        """
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'category' property that reflects the collection's data category."
            )

        return cls.Config.category

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        """What: Return the Qdrant collection name declared on this document class.
        Why: Every Qdrant API call needs the collection name as a string. Reading it
        from Config.name keeps it in one place — changing a collection name only
        requires editing the Config class, not every method that calls Qdrant.
        Raises ImproperlyConfigured if Config.name is missing to catch mistakes early.
        """
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            raise ImproperlyConfigured(
                "The class should define a Config class with" "the 'name' property that reflects the collection's name."
            )

        return cls.Config.name

    @classmethod
    def get_use_vector_index(cls: Type[T]) -> bool:
        """What: Return whether this document class should have a vector index in Qdrant.
        Why: Defaults to True (safe default) if Config.use_vector_index is not declared.
        This lets a subclass opt out of vector indexing by setting use_vector_index=False
        in its Config — used for cleaned documents and raw chunks that have no embeddings yet.
        """
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_vector_index"):
            # What: Default to True if the subclass didn't declare the flag.
            # Why: For embedded chunks (the most common production case) the index
            #      is required; defaulting to True is the safer choice.
            return True

        return cls.Config.use_vector_index

    @classmethod
    def group_by_class(
        cls: Type["VectorBaseDocument"], documents: list["VectorBaseDocument"]
    ) -> Dict["VectorBaseDocument", list["VectorBaseDocument"]]:
        """What: Group a flat list of documents into a dict keyed by their Python class.
        Why: Used in load_to_vector_db so documents of different types (PostChunk,
        ArticleChunk, RepositoryChunk) are routed to their own Qdrant collections
        in a single pass, without manual type checks in the pipeline step.
        """
        return cls._group_by(documents, selector=lambda doc: doc.__class__)

    @classmethod
    def group_by_category(cls: Type[T], documents: list[T]) -> Dict[DataCategory, list[T]]:
        """What: Group documents into a dict keyed by their DataCategory enum value.
        Why: Useful when the grouping key should be the semantic category (POSTS, ARTICLES,
        REPOSITORIES) rather than the exact Python class — e.g. when building
        per-category statistics or applying category-level pipeline logic.
        """
        return cls._group_by(documents, selector=lambda doc: doc.get_category())

    @classmethod
    def _group_by(cls: Type[T], documents: list[T], selector: Callable[[T], Any]) -> Dict[Any, list[T]]:
        """What: Generic grouping helper that partitions a list by a key derived from each element.
        Why: Avoids duplicating the group-by loop in group_by_class and group_by_category.
        The selector callable determines the grouping key, keeping the logic flexible
        without coupling it to a specific attribute.
        """
        grouped = {}
        for doc in documents:
            # What: Compute the grouping key for this document using the selector.
            # Why: The selector can be any callable (e.g. lambda doc: doc.__class__),
            #      making this method reusable for any grouping strategy.
            key = selector(doc)

            # What: Initialise the list for this key on first encounter.
            # Why: Avoids a KeyError when appending to a key that hasn't been seen yet.
            if key not in grouped:
                grouped[key] = []

            # What: Append the document to its group.
            # Why: Accumulates all documents sharing the same key into one list.
            grouped[key].append(doc)

        return grouped

    @classmethod
    def collection_name_to_class(cls: Type["VectorBaseDocument"], collection_name: str) -> type["VectorBaseDocument"]:
        """What: Walk the VectorBaseDocument class hierarchy to find the subclass whose
        Config.name matches `collection_name`.
        Why: Needed when deserialising data from Qdrant where we only know the collection
        name (a string) and need to reconstruct the correct typed model class.
        Recursively searches nested subclasses so deeply inherited types are found too.
        Raises ValueError with a clear message if no match is found.
        """
        for subclass in cls.__subclasses__():
            try:
                # What: Check if this direct subclass owns the requested collection name.
                # Why: get_collection_name() raises ImproperlyConfigured for abstract classes
                #      that don't define Config.name — the try/except skips those safely.
                if subclass.get_collection_name() == collection_name:
                    return subclass
            except ImproperlyConfigured:
                pass

            try:
                # What: Recurse into this subclass's own subclasses.
                # Why: The hierarchy can be multiple levels deep (e.g. Chunk → ArticleChunk);
                #      recursion ensures we find matches at any level.
                return subclass.collection_name_to_class(collection_name)
            except ValueError:
                # What: ValueError means no match in this branch — continue to next subclass.
                # Why: We only raise ValueError at the end when all branches are exhausted.
                continue

        # What: No subclass matched — raise with a descriptive message.
        # Why: Failing loudly here is better than returning None and causing a confusing
        #      AttributeError or silent data loss downstream.
        raise ValueError(f"No subclass found for collection name: {collection_name}")

    @classmethod
    def _has_class_attribute(cls: Type[T], attribute_name: str) -> bool:
        """What: Check whether `attribute_name` is declared as a field on this class
        or any of its base classes, walking the MRO (Method Resolution Order).
        Why: Used in from_record to decide whether to attach the `embedding` vector
        to the reconstructed model. Checking __annotations__ covers Pydantic field
        declarations which don't appear as instance attributes until validation runs.
        """
        # What: Check if the attribute is declared directly on this class.
        # Why: __annotations__ contains all Pydantic field declarations at the class level.
        if attribute_name in cls.__annotations__:
            return True

        # What: Recursively check each direct base class.
        # Why: The field may be declared in a parent class (e.g. EmbeddedChunk declares
        #      `embedding`; EmbeddedArticleChunk inherits it). Walking __bases__ ensures
        #      we find inherited fields at any depth in the hierarchy.
        for base in cls.__bases__:
            if hasattr(base, "_has_class_attribute") and base._has_class_attribute(attribute_name):
                return True

        # What: Return False if the attribute was not found anywhere in the hierarchy.
        # Why: from_record uses this to decide whether to set `embedding` — a False
        #      result means the class has no embedding field and we should skip it.
        return False
