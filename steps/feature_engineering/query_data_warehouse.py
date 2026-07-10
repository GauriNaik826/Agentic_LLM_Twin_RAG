# ThreadPoolExecutor: runs multiple I/O tasks concurrently in separate threads.
# as_completed: yields futures in the order they finish (not in submission order),
# so we can process each DB result as soon as it's ready rather than waiting for all.
from concurrent.futures import ThreadPoolExecutor, as_completed

# loguru logger for structured, readable log output throughout the step.
from loguru import logger
# Annotated lets us attach metadata (here, a name label) to a return type hint,
# which ZenML uses to name the output artifact in its tracking UI.
from typing_extensions import Annotated
# get_step_context: access the current ZenML step's runtime context (e.g., to attach metadata).
# step: decorator that registers this function as an executable ZenML pipeline step.
from zenml import get_step_context, step

# Utility helpers (e.g., splitting "First Last" into two separate name strings).
from llm_engineering.application import utils
# Base class for all NoSQL document models — used as a generic return type hint.
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
# Domain document models representing each content type stored in the data warehouse.
from llm_engineering.domain.documents import ArticleDocument, Document, PostDocument, RepositoryDocument, UserDocument


# Register this function as a ZenML step. ZenML will track its inputs, outputs,
# execution logs, and artifacts automatically when the pipeline runs.
@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:  # The output list will be named "raw_documents" in ZenML's artifact store.
    documents = []  # Accumulates all fetched documents across all authors.
    authors = []    # Accumulates UserDocument objects (reserved for potential downstream use).

    for author_full_name in author_full_names:
        logger.info(f"Querying data warehouse for user: {author_full_name}")

        # Split "John Doe" → ("John", "Doe") so we can look up the user by structured name fields.
        first_name, last_name = utils.split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")

        # Retrieve the user record from the DB if it exists, or create a new one if not.
        # This ensures the pipeline is idempotent — safe to re-run without duplicating users.
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        # Fetch all content types (articles, posts, repos) for this user concurrently.
        # Returns a dict like {"articles": [...], "posts": [...], "repositories": [...]}.
        results = fetch_all_data(user)

        # Flatten the dict of lists into a single list of documents for this user.
        user_documents = [doc for query_result in results.values() for doc in query_result]

        # Add this user's documents to the global list shared across all authors.
        documents.extend(user_documents)

    # Access ZenML's runtime step context to attach custom metadata to the output artifact.
    step_context = get_step_context()
    # Attach summary statistics (counts per collection, author names) to the "raw_documents"
    # artifact in ZenML's UI — useful for auditing what was fetched without opening the data.
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents  # Return the full flat list of raw documents to the next pipeline step.


def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)  # Convert the user's ObjectId to a plain string for DB query filtering.

    # Use a thread pool to fire all three DB queries (articles, posts, repos) simultaneously.
    # This is much faster than running them sequentially — especially when each query hits
    # a remote NoSQL database with network latency.
    with ThreadPoolExecutor() as executor:
        # Submit all three fetch functions concurrently. Each returns a Future object.
        # The dict maps each Future → its human-readable query name for error reporting.
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }

        results = {}
        # Iterate over futures as each one completes (whichever finishes first).
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]  # Look up which query this future belongs to.
            try:
                results[query_name] = future.result()  # Extract the returned list of documents.
            except Exception:
                # Log the full traceback but don't crash the whole step — just store an empty list
                # so the pipeline can continue with whatever data was successfully retrieved.
                logger.exception(f"'{query_name}' request failed.")
                results[query_name] = []

    return results  # e.g. {"articles": [...], "posts": [...], "repositories": [...]}


# Private helper: fetch all articles written by this user from the data warehouse.
def __fetch_articles(user_id) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)


# Private helper: fetch all social/blog posts written by this user.
def __fetch_posts(user_id) -> list[NoSQLBaseDocument]:
    return PostDocument.bulk_find(author_id=user_id)


# Private helper: fetch all code repositories associated with this user.
def __fetch_repositories(user_id) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)


# Build a summary metadata dict from the fetched documents.
# This is attached to the ZenML artifact for observability — so you can inspect
# what was fetched from ZenML's dashboard without reading the raw artifact data.
def _get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),  # Total count across all content types and authors.
    }
    for document in documents:
        collection = document.get_collection_name()  # e.g. "articles", "posts", "repositories"

        # Lazily initialise a sub-dict for this collection if it doesn't exist yet.
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()

        # Increment the per-collection document count.
        metadata[collection]["num_documents"] = metadata[collection].get("num_documents", 0) + 1
        # Record which author this document belongs to (may have duplicates at this stage).
        metadata[collection]["authors"].append(document.author_full_name)

    # Deduplicate author names per collection using a set conversion,
    # so the metadata shows unique contributors rather than one entry per document.
    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
