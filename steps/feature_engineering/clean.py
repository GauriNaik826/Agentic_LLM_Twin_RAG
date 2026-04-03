# What: Import Annotated from typing_extensions for adding metadata to type hints.
# Why: Annotated lets us attach ZenML artifact names to function parameters and
#      return values so the pipeline can track inputs/outputs by name.
from typing_extensions import Annotated

# What: Import get_step_context (runtime context accessor) and step (decorator) from ZenML.
# Why: `step` marks this function as a ZenML pipeline step; `get_step_context` gives
#      access to the running step's metadata so we can attach extra info to outputs.
from zenml import get_step_context, step

# What: Import CleaningDispatcher, which routes each document to its type-specific cleaner.
# Why: Different document types (articles, posts, repos) need different cleaning logic;
#      the dispatcher abstracts that routing away from this step.
from llm_engineering.application.preprocessing import CleaningDispatcher

# What: Import the CleanedDocument domain model.
# Why: Used as the element type in _get_metadata so we get proper type checking and
#      access to domain methods like get_category() and author_full_name.
from llm_engineering.domain.cleaned_documents import CleanedDocument


# What: Declare this function as a ZenML pipeline step.
# Why: The @step decorator registers the function with ZenML so it participates in
#      pipeline orchestration, artifact versioning, and experiment tracking.
@step
def clean_documents(
    # What: Accept a list of raw documents; the Annotated string "raw_documents" names
    #       the artifact in ZenML's artifact store.
    # Why: Naming the artifact lets downstream steps and dashboards identify it clearly.
    documents: Annotated[list, "raw_documents"],
    # What: Return a list and label it "cleaned_documents" in ZenML's artifact store.
    # Why: The name ties this output to downstream steps that expect "cleaned_documents".
) -> Annotated[list, "cleaned_documents"]:

    # What: Initialize an empty list to accumulate the cleaned documents.
    # Why: We process documents one at a time and collect results before returning.
    cleaned_documents = []

    # What: Iterate over every raw document passed in.
    # Why: Each document must be cleaned individually because they may be different types.
    for document in documents:
        # What: Dispatch the document to the appropriate cleaner based on its type.
        # Why: CleaningDispatcher inspects the document's type and routes it to the
        #      correct cleaning strategy (e.g. HTML stripping, whitespace normalization).
        cleaned_document = CleaningDispatcher.dispatch(document)

        # What: Add the cleaned document to the results list.
        # Why: Accumulates all cleaned documents so they can be returned as a single artifact.
        cleaned_documents.append(cleaned_document)

    # What: Retrieve the ZenML step context for the currently running step.
    # Why: We need it to attach custom metadata to the output artifact so the pipeline
    #      dashboard shows useful statistics without having to open the artifact itself.
    step_context = get_step_context()

    # What: Attach the metadata dict (document counts & authors) to the "cleaned_documents" output.
    # Why: This makes the stats visible in ZenML's UI/tracking without loading the full artifact,
    #      enabling quick quality checks after each pipeline run.
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))

    # What: Return the list of cleaned documents as the step's output artifact.
    # Why: ZenML serializes and versions this list; the next step in the pipeline
    #      will receive it as its "cleaned_documents" input.
    return cleaned_documents


# What: Private helper that builds a summary metadata dict from the cleaned documents.
# Why: Separating metadata construction keeps clean_documents readable and makes the
#      logic independently testable.
def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:

    # What: Seed the metadata dict with the total document count.
    # Why: Provides an at-a-glance total across all categories in the ZenML dashboard.
    metadata = {"num_documents": len(cleaned_documents)}

    # What: Loop through each cleaned document to build per-category statistics.
    # Why: We want counts and author lists broken down by category (article, post, repo).
    for document in cleaned_documents:

        # What: Get the category string for this document (e.g. "articles", "posts").
        # Why: Used as the grouping key so stats are organised by document type.
        category = document.get_category()

        # What: Create an empty sub-dict for this category if it doesn't exist yet.
        # Why: Ensures we can safely write to metadata[category] on the next lines.
        if category not in metadata:
            metadata[category] = {}

        # What: Initialize an empty authors list for the category on first encounter.
        # Why: Ensures we can call .append() without a KeyError on subsequent documents.
        if "authors" not in metadata[category]:
            metadata[category]["authors"] = list()

        # What: Increment the per-category document count by 1.
        # Why: Tracks how many documents of each type were cleaned in this pipeline run.
        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1

        # What: Append this document's author name to the category's author list.
        # Why: Collects all authors so we can later deduplicate and report unique authors.
        metadata[category]["authors"].append(document.author_full_name)

    # What: Iterate over every top-level value in the metadata dict.
    # Why: We need to post-process each category sub-dict to deduplicate author names.
    for value in metadata.values():

        # What: Check that the value is a category sub-dict (not the top-level int count)
        #       and that it has an authors list.
        # Why: Guards against trying to call set() on the plain integer "num_documents" value.
        if isinstance(value, dict) and "authors" in value:

            # What: Replace the authors list with a deduplicated version using set conversion.
            # Why: The same author can appear across multiple documents; unique names give a
            #      clearer picture of contributor coverage in the pipeline run.
            value["authors"] = list(set(value["authors"]))

    # What: Return the fully assembled metadata dict.
    # Why: This dict is passed to ZenML's add_output_metadata so it appears in the UI.
    return metadata
