"""
nosql.py — Base class for all MongoDB-backed domain documents
-------------------------------------------------------------

WHY THIS FILE EXISTS
--------------------
Every domain document (UserDocument, ArticleDocument, RepositoryDocument …)
needs the same set of low-level MongoDB operations: insert, find, bulk-find,
upsert, and ID translation between Python's UUID and MongoDB's "_id" string.

Rather than duplicating that logic in every model, `NoSQLBaseDocument`
centralises it here. Concrete document classes only need to:
  1. Declare their fields (Pydantic).
  2. Provide an inner `Settings.name` pointing to their MongoDB collection.

Everything else — persistence, querying, serialisation — is inherited for free.

This is a classic "Active Record"-style pattern: the domain object knows how
to save and retrieve itself, keeping persistence logic close to the data model
without requiring a separate repository class for simple CRUD operations.
"""

import uuid            # Python's built-in UUID generation — used to create unique document IDs
from abc import ABC   # Makes NoSQLBaseDocument abstract so it can't be instantiated directly
from typing import Generic, Type, TypeVar  # Generic[T] allows methods to return the correct
                                           # subclass type instead of always returning the base type

from loguru import logger                  # Structured logging for all DB operation warnings/errors
from pydantic import UUID4, BaseModel, Field  # Pydantic BaseModel provides automatic field
                                              # validation; UUID4 enforces the v4 UUID type;
                                              # Field lets us configure default_factory
from pymongo import errors                 # Typed MongoDB error classes for targeted exception handling

# ImproperlyConfigured is a domain-level sentinel for misconfigured models
# (e.g., a document class that forgot to define Settings.name).
from llm_engineering.domain.exceptions import ImproperlyConfigured

# `connection` is a singleton MongoClient created by MongoDatabaseConnector.
# Importing it here means all document classes share one connection pool
# rather than opening a new connection per operation.
from llm_engineering.infrastructure.db.mongo import connection

# `settings` holds DATABASE_NAME (and all other config) loaded from .env or
# the ZenML secret store. Reading the DB name from settings keeps it
# configurable without touching source code.
from llm_engineering.settings import settings

# Obtain a handle to the application database at module load time.
# `_database[collection_name]` is then used inside every method to target the
# correct collection. Using a module-level variable avoids re-calling
# get_database() on every method invocation.
_database = connection.get_database(settings.DATABASE_NAME)


# TypeVar bound to NoSQLBaseDocument so that class methods declared with
# `cls: Type[T]` always return the *concrete* subclass (e.g., ArticleDocument),
# not the abstract base. Without this, `ArticleDocument.find(...)` would be
# typed as returning `NoSQLBaseDocument`, losing field information.
T = TypeVar("T", bound="NoSQLBaseDocument")


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    """Abstract base class providing MongoDB persistence for all domain documents.

    Inherits from Pydantic's BaseModel for field validation and serialisation,
    and from Generic[T] so that classmethods (find, get_or_create, etc.) can
    return properly typed instances of whichever concrete subclass is used.

    Subclasses must define an inner Settings class with a `name` attribute
    that specifies the target MongoDB collection, e.g.:

        class Settings:
            name = "articles"   # or DataCategory.ARTICLES

    The class is marked ABC to prevent accidental direct instantiation.
    """

    # Every document gets a UUID4 primary key generated automatically on
    # construction. Using uuid4 (random UUID) avoids collisions without
    # needing a central counter or database sequence.
    # `default_factory=uuid.uuid4` means a *new* UUID is generated each time
    # a document is created, rather than sharing one default value across all
    # instances (the classic mutable-default pitfall).
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        """Two documents are equal if and only if they are the same type and share the same id.

        Why check `isinstance` first?
        An ArticleDocument and a UserDocument could theoretically have the same
        UUID (astronomically unlikely, but possible). Requiring same class AND
        same id prevents false equality across different collections.
        """
        if not isinstance(value, self.__class__):
            return False

        # Compare only by id — the unique, stable identifier. Field values
        # may change (e.g., content updated), but the id never changes.
        return self.id == value.id

    def __hash__(self) -> int:
        """Make documents hashable so they can be stored in sets or used as dict keys.

        Pydantic models are not hashable by default because they are mutable.
        We override __hash__ based on id alone, consistent with __eq__.
        This is safe as long as callers don't mutate id (which they shouldn't).
        """
        return hash(self.id)

    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """Deserialise a raw MongoDB document dict into a typed domain object.

        MongoDB stores the primary key as "_id" (a string). Pydantic models
        use "id" (a UUID4). This method bridges that gap by:
          1. Popping "_id" from the dict (mutates the input — intentional, as
             the caller doesn't need the raw dict afterwards).
          2. Passing the remaining fields plus `id=_id` to the Pydantic
             constructor, which validates and coerces all field types.

        Why `data.pop("_id")` instead of `data.get`?
        pop removes "_id" so it doesn't collide with Pydantic's "id" field
        when we unpack the dict with `**data`.
        """
        if not data:
            # An empty dict means MongoDB returned nothing or the document is
            # corrupted. Fail loudly rather than silently constructing an
            # empty object with no meaningful fields.
            raise ValueError("Data is empty.")

        # Remove and capture the "_id" field. MongoDB guarantees this key
        # exists on every stored document.
        id = data.pop("_id")

        # Merge remaining fields with the renamed id and hand off to Pydantic
        # for validation. dict(data, id=id) is equivalent to {**data, "id": id}.
        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """Serialise this Pydantic model into a dict ready for MongoDB insertion.

        MongoDB expects "_id" (not "id") as the primary key field. This method:
          1. Delegates to `model_dump` to get a plain Python dict (UUIDs already
             converted to strings by our overridden model_dump).
          2. Renames "id" → "_id" so PyMongo stores it as the document key.
          3. Converts any remaining UUID values to strings as a safety net
             (in case model_dump missed any nested UUIDs).
        """
        # `exclude_unset=False` (default): include all fields, even those that
        # weren't explicitly set by the caller, so MongoDB sees a complete document.
        exclude_unset = kwargs.pop("exclude_unset", False)

        # `by_alias=True` (default): use Pydantic field aliases (e.g., "author_id")
        # instead of the Python attribute name when serialising. This ensures
        # MongoDB keys match what the rest of the pipeline expects.
        by_alias = kwargs.pop("by_alias", True)

        # Delegate to our overridden model_dump (which stringifies UUIDs).
        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        # Rename "id" → "_id". The check avoids a KeyError if "_id" was already
        # injected somehow (e.g., a subclass added it manually).
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        # Belt-and-suspenders: stringify any UUID fields that model_dump may
        # have missed (e.g., UUIDs inside nested dicts not handled by the loop
        # in model_dump). PyMongo cannot serialise uuid.UUID objects natively.
        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed

    def model_dump(self: T, **kwargs) -> dict:
        """Override Pydantic's model_dump to stringify all UUID fields.

        Why override?
        Pydantic v2's default model_dump returns UUID objects as-is. MongoDB's
        PyMongo driver cannot serialise `uuid.UUID` objects. By converting them
        to strings here, every path that calls model_dump (including to_mongo)
        gets safe, serialisable output automatically.
        """
        # Call the parent Pydantic model_dump to get the base dict with all
        # fields serialised according to Pydantic's rules.
        dict_ = super().model_dump(**kwargs)

        # Walk every top-level field and convert UUID instances to strings.
        # Nested UUIDs (inside dicts/lists) are handled separately in to_mongo.
        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)

        return dict_

    def save(self: T, **kwargs) -> T | None:
        """Insert this document into its MongoDB collection.

        Returns `self` on success so callers can chain:
            instance = MyDoc(...).save()

        Returns `None` on a write error (e.g., duplicate key, schema
        validation failure at the MongoDB level) so the pipeline can
        detect failure without crashing.
        """
        # Resolve the collection name from the subclass's Settings.name and
        # obtain a PyMongo Collection object.
        collection = _database[self.get_collection_name()]
        try:
            # `insert_one` sends a single document to MongoDB. We convert self
            # to a mongo-safe dict first via to_mongo().
            collection.insert_one(self.to_mongo(**kwargs))

            # Return self so callers can do `saved = document.save()` and
            # still have access to the full document object.
            return self
        except errors.WriteError:
            # WriteError covers constraint violations and schema mismatches.
            # We log with `exception` (which includes the full stack trace)
            # and return None rather than re-raising, allowing the pipeline
            # to mark the item as failed and continue.
            logger.exception("Failed to insert document.")

            return None

    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:
        """Fetch an existing document matching `filter_options`, or create one.

        This is the standard upsert pattern:
          - If a matching document exists → deserialise and return it.
          - If not → create a new instance from the filter fields and save it.

        Use case: the ETL pipeline calls `UserDocument.get_or_create(first_name=...,
        last_name=...)` so users are never duplicated across pipeline runs.

        Raises `errors.OperationFailure` (re-raised after logging) if MongoDB
        reports a query-level failure (e.g., auth error, network partition).
        """
        collection = _database[cls.get_collection_name()]
        try:
            # `find_one` returns the first document matching the filter dict,
            # or None if no match is found.
            instance = collection.find_one(filter_options)
            if instance:
                # Document already exists — deserialise and return it.
                return cls.from_mongo(instance)

            # No existing document — create a new one from the filter fields.
            # This assumes filter_options contains all required fields; if not,
            # Pydantic will raise a ValidationError here.
            new_instance = cls(**filter_options)

            # Persist and reassign — save() returns self (or None on failure).
            new_instance = new_instance.save()

            return new_instance
        except errors.OperationFailure:
            # OperationFailure signals a MongoDB-level query problem (not a
            # "document not found" — that is handled by the None check above).
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")

            # Re-raise so the caller can decide whether to retry or abort.
            raise

    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        """Insert a list of documents into MongoDB in a single network round-trip.

        Why bulk_insert instead of calling save() in a loop?
        `insert_many` sends all documents in one batch, which is significantly
        faster than N individual `insert_one` calls, each with their own
        network overhead. This is used during the feature-engineering pipeline
        when thousands of chunks are written at once.

        Returns True on success, False on any write failure so the caller can
        log or retry without the pipeline crashing.
        """
        collection = _database[cls.get_collection_name()]
        try:
            # Generator expression: convert each document to a mongo-safe dict
            # lazily, avoiding building the entire list in memory first.
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)

            return True
        except (errors.WriteError, errors.BulkWriteError):
            # BulkWriteError is specific to insert_many — it contains details
            # about which individual documents failed. We log at error level
            # (not exception) because bulk failures are often expected
            # (e.g., duplicate IDs on re-runs with ordered=True).
            logger.error(f"Failed to insert documents of type {cls.__name__}")

            return False

    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:
        """Find a single document matching `filter_options`.

        Used extensively by crawlers to check deduplication before re-fetching
        a URL: `ArticleDocument.find(link=link)`.

        Returns a fully-typed domain object, or None if no match found.
        Returning None (rather than raising) keeps caller code simple:
            if existing := ArticleDocument.find(link=link): return
        """
        collection = _database[cls.get_collection_name()]
        try:
            # `find_one` returns the first match or None — exactly the
            # semantics we want for a "does this document already exist?" check.
            instance = collection.find_one(filter_options)
            if instance:
                # Convert raw dict → typed Pydantic model.
                return cls.from_mongo(instance)

            # No document found — return None to signal "not present".
            return None
        except errors.OperationFailure:
            # Log at error (not exception) because "failed to retrieve" is
            # recoverable in most pipeline contexts; we return None so callers
            # treat this as "not found" and carry on.
            logger.error("Failed to retrieve document")

            return None

    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        """Fetch all documents matching `filter_options` as a typed list.

        Used during feature engineering to load all raw documents of a given
        type for batch processing (cleaning, chunking, embedding).

        Uses a list comprehension with a walrus-operator guard to skip any
        raw documents that fail deserialisation (from_mongo returns None-like),
        ensuring a partial failure doesn't abort the entire batch.
        """
        collection = _database[cls.get_collection_name()]
        try:
            # `find` returns a cursor (lazy iterator) over all matching docs.
            instances = collection.find(filter_options)

            # Iterate the cursor and deserialise each raw dict into a typed
            # model. The walrus operator `:=` assigns and tests in one step:
            # if from_mongo returns a falsy value (shouldn't happen, but
            # defensive), that entry is excluded from the result list.
            return [document for instance in instances if (document := cls.from_mongo(instance)) is not None]
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")

            # Return an empty list rather than None so callers can always
            # iterate the result without an extra None check.
            return []

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        """Return the MongoDB collection name declared on this document class.

        Every concrete subclass must define:
            class Settings:
                name = "articles"   # or a DataCategory enum value

        This method enforces that contract at runtime. Failing early with a
        clear `ImproperlyConfigured` message is far easier to debug than a
        cryptic KeyError or a document silently being written to the wrong
        collection (or no collection at all).
        """
        # hasattr checks protect against subclasses that omitted Settings
        # entirely or defined Settings without the `name` attribute.
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )

        # Return the value — could be a plain string or a DataCategory StrEnum;
        # PyMongo accepts both because StrEnum subclasses str.
        return cls.Settings.name
