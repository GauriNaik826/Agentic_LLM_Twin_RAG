# What: Import Literal for a constrained set of valid route values.
# Why: Keeps route outputs type-safe and easier to validate.
from typing import Literal

# What: Import prompt builder for structured chat messages.
# Why: The classifier LLM expects system/human role-formatted input.
from langchain_core.prompts import ChatPromptTemplate
# What: Import OpenAI chat model wrapper.
# Why: Used to classify user intent into a routing label.
from langchain_openai import ChatOpenAI

# What: Import application settings singleton.
# Why: Provides model ID and API key from environment/ZenML settings.
from llm_engineering.settings import settings

# What: Define allowed route labels.
# Why: Ensures router outputs are one of the supported downstream branches.
Route = Literal["rag", "web", "twin_writer"]


# What: Query router component.
# Why: Separates route classification from supervisor execution logic.
class QueryRouter:
    """Simple LLM-based query router for supervisor orchestration."""

    # What: Initialize reusable classifier prompt template.
    # Why: Building once avoids repeating prompt construction per request.
    def __init__(self) -> None:
        # What: Create chat-style prompt with explicit routing instructions.
        # Why: Constrains LLM output to route tokens we can map in the graph.
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    # What: System instruction message.
                    # Why: Defines strict classification behavior and route semantics.
                    "system",
                    (
                        "Classify the user query into exactly one route: rag, web, or twin_writer. "
                        "Return only one token from this set with no extra text. "
                        "Use web for latest news/current events/time-sensitive questions. "
                        "Use rag for questions needing author knowledge from stored documents. "
                        "Use twin_writer for style/tone writing requests."
                    ),
                ),
                # What: Human message with query placeholder.
                # Why: Injects the runtime user query into the prompt.
                ("human", "{query}"),
            ]
        )

    # What: Classify a user query into a route label.
    # Why: Supervisor uses this decision to choose the execution branch.
    def classify(self, query: str) -> Route:
        # Keep routing robust in environments where OpenAI is not configured.
        # What: If API key is missing, skip LLM call.
        # Why: Guarantees deterministic fallback route and avoids runtime failure.
        if settings.OPENAI_API_KEY is None:
            return "rag"

        # What: Instantiate classifier model.
        # Why: Uses configured model/credentials and deterministic temperature.
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

        # What: Run LLM classification with formatted prompt.
        # Why: Produces route token from natural-language query.
        response = llm.invoke(self._prompt.format_messages(query=query))
        # What: Normalize output text.
        # Why: Makes route matching robust to whitespace/case variance.
        route = response.content.strip().lower()

        # What: Validate model output against known routes.
        # Why: Protects graph routing from unexpected/invalid responses.
        if route not in {"rag", "web", "twin_writer"}:
            return "rag"

        # What: Return validated route label.
        # Why: Downstream conditional edge expects one of these tokens.
        return route