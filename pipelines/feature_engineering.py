# Import the @pipeline decorator from ZenML — this marks the function as a ZenML pipeline,
# enabling orchestration, caching, artifact tracking, and step dependency resolution.
from zenml import pipeline

# Import the feature engineering steps module, which contains individual pipeline steps
# such as querying the data warehouse, cleaning, chunking, embedding, and loading to a vector DB.
from steps import feature_engineering as fe_steps


# Decorate this function as a ZenML pipeline. ZenML will treat each called function
# inside as a "step", automatically wiring inputs/outputs and managing execution order.
@pipeline
def feature_engineering(author_full_names: list[str], wait_for: str | list[str] | None = None) -> list[str]:
    # Step 1: Fetch raw documents from the data warehouse for the given authors.
    # `after=wait_for` ensures this step only runs after a preceding pipeline step
    # (identified by its invocation ID) has completed — used to chain pipelines together.
    raw_documents = fe_steps.query_data_warehouse(author_full_names, after=wait_for)

    # Step 2: Clean the raw documents (e.g., remove noise, normalize text, deduplicate).
    # The output is a collection of cleaned document artifacts.
    cleaned_documents = fe_steps.clean_documents(raw_documents)

    # Step 3a: Load the cleaned (but not yet embedded) documents directly into the vector DB.
    # This stores the plain-text versions for retrieval or future reprocessing.
    # `last_step_1` holds the ZenML step invocation object, used later to get its ID.
    last_step_1 = fe_steps.load_to_vector_db(cleaned_documents)

    # Step 3b: Chunk the cleaned documents into smaller passages and generate vector embeddings.
    # Chunking makes long documents indexable; embeddings enable semantic similarity search.
    embedded_documents = fe_steps.chunk_and_embed(cleaned_documents)

    # Step 4: Load the embedded (chunked + vectorized) documents into the vector DB.
    # This is the primary data that will be queried during RAG (Retrieval-Augmented Generation).
    # `last_step_2` holds the ZenML step invocation object for its ID.
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

    # Return the invocation IDs of the two final load steps.
    # These IDs can be passed to downstream pipelines via `wait_for`, ensuring
    # those pipelines only start after both loads have successfully completed.
    return [last_step_1.invocation_id, last_step_2.invocation_id]
