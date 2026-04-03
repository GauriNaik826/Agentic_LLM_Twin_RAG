# What: Import Annotated from typing_extensions for attaching metadata to type hints.
# Why: Allows us to label ZenML input/output artifacts by name so the pipeline UI
#      can display and track them clearly.
from typing_extensions import Annotated

# What: Import get_step_context (runtime context accessor) and step (decorator) from ZenML.
# Why: `step` registers this function as a ZenML pipeline step; `get_step_context`
#      lets us attach custom metadata to output artifacts during execution.
from zenml import get_step_context, step

# What: Import the internal utils module containing miscellaneous helpers (e.g. batch).
# Why: We use utils.misc.batch() to split chunks into fixed-size groups before
#      embedding, which prevents memory spikes and respects API rate limits.
from llm_engineering.application import utils

# What: Import the dispatcher classes that route chunking and embedding by document type.
# Why: Different document types require different chunking strategies; both dispatchers
#      encapsulate that routing logic so this step stays clean and type-agnostic.
from llm_engineering.application.preprocessing import ChunkingDispatcher, EmbeddingDispatcher

# What: Import the Chunk domain model representing a single text chunk.
# Why: Used as the element type for _add_chunks_metadata so we get type-checked
#      access to chunk attributes like get_category() and author_full_name.
from llm_engineering.domain.chunks import Chunk

# What: Import the EmbeddedChunk domain model — a chunk that also carries a vector embedding.
# Why: Used as the element type for _add_embeddings_metadata and as the final
#      output artifact that gets stored in the vector database.
from llm_engineering.domain.embedded_chunks import EmbeddedChunk


# What: Register this function as a ZenML pipeline step.
# Why: The decorator hooks the function into ZenML's orchestration, artifact versioning,
#      and experiment-tracking infrastructure.
@step
def chunk_and_embed(
    # What: Accept the list of cleaned documents produced by the previous step;
    #       "cleaned_documents" is the artifact name ZenML uses to track it.
    # Why: Naming the artifact creates an explicit lineage link between the cleaning
    #      step and this step in the ZenML dashboard.
    cleaned_documents: Annotated[list, "cleaned_documents"],
    # What: Return the list of embedded chunks labelled "embedded_documents".
    # Why: The label lets the next step (e.g. vector DB loader) reference this
    #      artifact by name and keeps the lineage graph readable.
) -> Annotated[list, "embedded_documents"]:

    # What: Initialise the metadata dict with empty sub-dicts for chunking/embedding
    #       stats and a top-level total document count.
    # Why: Pre-seeding the keys avoids KeyError checks later and gives a clear
    #      structure that will be surfaced in ZenML's artifact metadata panel.
    metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)}

    # What: Start with an empty list that will hold all embedded chunks.
    # Why: We accumulate results across all documents before returning them as
    #      one versioned artifact.
    embedded_chunks = []

    # What: Iterate over each cleaned document.
    # Why: Every document is processed independently because it may be a different
    #      type (article, post, repo) with its own chunking strategy.
    for document in cleaned_documents:

        # What: Split the document into text chunks using the appropriate strategy
        #       for its type (dispatched internally by ChunkingDispatcher).
        # Why: LLMs have context-length limits; chunking breaks long documents into
        #      pieces that fit within those limits and improve retrieval precision.
        chunks = ChunkingDispatcher.dispatch(document)

        # What: Accumulate per-category chunk statistics into the metadata dict.
        # Why: Tracks how many chunks were produced per category so we can audit
        #      chunking quality in the ZenML UI after the run.
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        # What: Iterate over the chunks in groups of 10.
        # Why: Batching controls memory usage and respects the rate limits or
        #      maximum batch sizes imposed by the embedding model / API.
        for batched_chunks in utils.misc.batch(chunks, 10):

            # What: Embed the current batch of chunks into dense vector representations.
            # Why: Vector embeddings are what gets stored in the vector DB and queried
            #      at retrieval time; the dispatcher picks the right embedding model/strategy.
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)

            # What: Add all embedded chunks from this batch to the running list.
            # Why: extend() appends each element of the batch rather than nesting the
            #      batch as a sub-list, keeping embedded_chunks flat.
            embedded_chunks.extend(batched_embedded_chunks)

    # What: Compute and store per-category embedding statistics into the metadata dict.
    # Why: Lets us verify that every chunk was successfully embedded before the data
    #      is pushed to the vector store.
    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])

    # What: Record the total number of chunks produced across all documents.
    # Why: Provides a quick sanity-check count in the ZenML UI to spot unexpected
    #      drops (e.g. a document that produced zero chunks).
    metadata["num_chunks"] = len(embedded_chunks)

    # What: Record the total number of successfully embedded chunks.
    # Why: Should match num_chunks; a mismatch would indicate embedding failures
    #      that need investigation before loading data into the vector DB.
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    # What: Retrieve the ZenML step context for the currently running step.
    # Why: Required to call add_output_metadata and attach our stats dict to the
    #      output artifact rather than just printing or logging it.
    step_context = get_step_context()

    # What: Attach the complete metadata dict to the "embedded_documents" output artifact.
    # Why: Makes chunking and embedding statistics visible in ZenML's dashboard without
    #      having to load and inspect the full artifact.
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)

    # What: Return the flat list of embedded chunks as this step's output artifact.
    # Why: ZenML serializes and versions the list; the downstream vector-DB loading
    #      step receives it as its input.
    return embedded_chunks


# What: Private helper that merges chunk-level statistics into an existing metadata dict.
# Why: Separating this logic keeps chunk_and_embed readable and makes the aggregation
#      logic independently testable.
def _add_chunks_metadata(chunks: list[Chunk], metadata: dict) -> dict:

    # What: Loop over each chunk to gather per-category statistics.
    # Why: Statistics need to be broken down by category (article, post, repo) so
    #      we can spot imbalances in the data across document types.
    for chunk in chunks:

        # What: Retrieve the category string (e.g. "articles") for this chunk.
        # Why: Used as the grouping key in the metadata dict.
        category = chunk.get_category()

        # What: On the first chunk for a category, seed its sub-dict with the
        #       chunk's own metadata (e.g. source URL, document ID).
        # Why: Captures category-level context from the first chunk seen; subsequent
        #      chunks just update the counts and authors list.
        if category not in metadata:
            metadata[category] = chunk.metadata

        # What: Ensure the authors list exists before appending.
        # Why: chunk.metadata may not include an "authors" key, so we initialise
        #      it explicitly to avoid a KeyError on .append().
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        # What: Increment the running chunk count for this category.
        # Why: Tracks total chunks per category for quality monitoring.
        metadata[category]["num_chunks"] = metadata[category].get("num_chunks", 0) + 1

        # What: Append this chunk's author name to the category's authors list.
        # Why: Collects all authors so we can deduplicate them after the loop.
        metadata[category]["authors"].append(chunk.author_full_name)

    # What: Deduplicate the authors list for every category sub-dict.
    # Why: The same author appears on every chunk of their documents; set conversion
    #      gives a clean list of unique contributors.
    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    # What: Return the updated metadata dict.
    # Why: The caller merges this back into the top-level metadata so chunking stats
    #      are available when add_output_metadata is called.
    return metadata


# What: Private helper that merges embedding-level statistics into an existing metadata dict.
# Why: Keeps chunk_and_embed clean and mirrors the pattern used for chunk metadata,
#      making the codebase consistent and easy to follow.
def _add_embeddings_metadata(embedded_chunks: list[EmbeddedChunk], metadata: dict) -> dict:

    # What: Loop over every embedded chunk to build per-category embedding stats.
    # Why: Allows per-category auditing — e.g. confirming all article chunks were
    #      embedded — rather than just a single global count.
    for embedded_chunk in embedded_chunks:

        # What: Get the category string for this embedded chunk.
        # Why: Used as the grouping key to organise stats by document type.
        category = embedded_chunk.get_category()

        # What: Seed the category sub-dict with the embedded chunk's metadata on
        #       first encounter for that category.
        # Why: Captures source metadata (model name, dimensions, etc.) stored on
        #      the embedded chunk for later inspection in the ZenML UI.
        if category not in metadata:
            metadata[category] = embedded_chunk.metadata

        # What: Ensure the authors list exists for this category.
        # Why: embedded_chunk.metadata may not contain "authors", so we initialise
        #      it to avoid a KeyError on .append().
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        # What: Append the author of this embedded chunk to the category's list.
        # Why: Collects authors across all embedded chunks for deduplication after
        #      the loop.
        metadata[category]["authors"].append(embedded_chunk.author_full_name)

    # What: Deduplicate the authors list for every category that has one.
    # Why: Multiple embedded chunks share the same author; unique names give an
    #      accurate contributor count in the ZenML metadata panel.
    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    # What: Return the updated embedding metadata dict.
    # Why: The caller stores it under metadata["embedding"] so it appears alongside
    #      chunking stats when ZenML records the output artifact.
    return metadata
