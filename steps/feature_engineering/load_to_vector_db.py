# What: Import the loguru logger for structured, levelled log output.
# Why: Provides clear runtime visibility (info, error) without needing to configure
#      Python's standard logging module; output is easy to grep in CI/CD pipelines.
from loguru import logger

# What: Import Annotated from typing_extensions for attaching metadata to type hints.
# Why: Lets us label ZenML artifacts by name so the pipeline UI tracks inputs/outputs
#      with human-readable identifiers rather than positional indices.
from typing_extensions import Annotated

# What: Import the ZenML step decorator.
# Why: Registers this function as a ZenML pipeline step, enabling artifact versioning,
#      orchestration, and experiment tracking automatically.
from zenml import step

# What: Import the internal utils module which provides miscellaneous helpers.
# Why: We specifically use utils.misc.batch() to split documents into fixed-size
#      groups, preventing memory spikes and keeping DB write operations manageable.
from llm_engineering.application import utils

# What: Import VectorBaseDocument, the base class for all vector-store document types.
# Why: Its group_by_class() and bulk_insert() class methods abstract away the
#      type-specific collection routing so this step stays generic.
from llm_engineering.domain.base import VectorBaseDocument


# What: Register this function as a ZenML pipeline step.
# Why: The decorator integrates the function into ZenML's orchestration engine,
#      giving us free artifact versioning and pipeline lineage tracking.
@step
def load_to_vector_db(
    # What: Accept a flat list of embedded documents; "documents" is the artifact name
    #       ZenML uses to identify this input in its lineage graph.
    # Why: Naming the artifact creates an explicit dependency link from the embedding
    #      step to this step in the ZenML UI.
    documents: Annotated[list, "documents"],
    # What: Return a boolean success flag labelled "successful" as the output artifact.
    # Why: Downstream steps or monitoring can check this flag to decide whether to
    #      continue the pipeline or trigger an alert on failure.
) -> Annotated[bool, "successful"]:

    # What: Log an info message showing how many documents are about to be loaded.
    # Why: Provides an immediate audit trail in logs so operators can verify the
    #      expected volume reaches the vector DB at runtime.
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    # What: Group the flat document list into a dict keyed by their Python class type.
    # Why: Different document types (e.g. ArticleEmbeddedChunk, PostEmbeddedChunk)
    #      map to different vector-DB collections; grouping lets us route each batch
    #      to the correct collection without manual type checks.
    grouped_documents = VectorBaseDocument.group_by_class(documents)

    # What: Iterate over each document class and its corresponding list of documents.
    # Why: Each class targets a different collection, so we must process them
    #      separately to ensure writes go to the right place.
    for document_class, documents in grouped_documents.items():

        # What: Log which collection is about to receive documents.
        # Why: Makes it easy to trace which collection was being written to if an
        #      error occurs, speeding up debugging.
        logger.info(f"Loading documents into {document_class.get_collection_name()}")

        # What: Split the documents for this class into batches of 4.
        # Why: Writing in small batches limits the payload size per request, avoids
        #      hitting vector-DB write limits, and reduces memory pressure.
        for documents_batch in utils.misc.batch(documents, size=4):

            # What: Attempt to bulk-insert the current batch into the vector DB.
            # Why: bulk_insert is more efficient than individual inserts because it
            #      reduces round-trip overhead and leverages the DB's batch API.
            try:
                document_class.bulk_insert(documents_batch)

            # What: Catch any exception raised during the insert.
            # Why: A broad catch ensures transient errors (network timeouts, schema
            #      mismatches) don't silently corrupt partial state; we log and abort.
            except Exception:

                # What: Log an error message naming the collection that failed.
                # Why: Pinpoints exactly which collection caused the failure so the
                #      operator knows where to investigate without reading a full stack trace.
                logger.error(f"Failed to insert documents into {document_class.get_collection_name()}")

                # What: Return False to signal that the step did not complete successfully.
                # Why: Returning a boolean failure flag lets the ZenML pipeline and any
                #      monitoring layer detect the problem and halt or alert rather than
                #      silently proceeding with incomplete data in the vector DB.
                return False

    # What: Return True once all documents across all collections have been inserted.
    # Why: Signals to ZenML and any downstream steps that the vector DB is fully
    #      populated and safe to query.
    return True
