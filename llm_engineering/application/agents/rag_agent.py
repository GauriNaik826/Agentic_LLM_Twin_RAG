# What: Postpone evaluation of annotations so forward refs and modern typing remain safe.
# Why: Avoids runtime issues with type hints in some import orders.
from __future__ import annotations

# What: Import project settings (env/config values).
# Why: Needed to resolve SageMaker endpoint name.
from llm_engineering import settings
# What: Import WebAgent for fallback behavior.
# Why: RAG can fall back to web when retrieval/generation quality is insufficient.
from llm_engineering.application.agents.web_agent import WebAgent
# What: Import circuit breaker primitives.
# Why: Protect external dependencies (retriever, SageMaker) from cascading failures.
from llm_engineering.application.guardrails import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
# What: Import retriever used by RAG.
# Why: Retrieves semantic chunks from the vector store.
from llm_engineering.application.rag.retriever import ContextRetriever
# What: Import chunk model utilities.
# Why: Used for typing and context serialization.
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
# What: Import inference helpers for SageMaker generation.
# Why: Sends final prompt to the tuned model endpoint.
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint


# What: Encapsulates RAG execution plus resilience/fallback policies.
# Why: Keeps retrieval logic reusable for supervisor graph nodes.
class RAGAgent:
    # What: Minimum number of retrieved docs required to trust RAG.
    # Why: Too few docs usually means weak evidence.
    MIN_DOCS = 2
    # What: Minimum acceptable similarity score for retrieved docs.
    # Why: Low scores indicate poor relevance.
    MIN_SIMILARITY_SCORE = 0.20
    # What: Circuit breaker for retriever dependency.
    # Why: Opens after repeated failures to avoid repeated slow/broken calls.
    _retriever_cb = CircuitBreaker(
        name="rag_retriever",
        config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30, half_open_max_calls=1),
    )
    # What: Circuit breaker for SageMaker generation dependency.
    # Why: Protects system from repeated model endpoint failures.
    _sagemaker_cb = CircuitBreaker(
        name="rag_sagemaker",
        config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30, half_open_max_calls=1),
    )

    # What: Remove duplicate retrieved chunks.
    # Why: Prevent repeated content from biasing generation and wasting context window.
    @staticmethod
    def _deduplicate_docs(documents: list[EmbeddedChunk]) -> list[EmbeddedChunk]:
        # What: Destination list for unique docs.
        # Why: Preserve original order while deduplicating.
        unique_docs: list[EmbeddedChunk] = []
        # What: Fingerprint set for quick duplicate checks.
        # Why: O(1) membership checks are efficient.
        seen: set[tuple[str, str, str]] = set()

        # What: Iterate each candidate chunk.
        # Why: Build a stable unique list.
        for doc in documents:
            # What: Build dedupe key from source platform/author/content.
            # Why: These dimensions capture practical duplicates.
            key = (doc.platform, doc.author_full_name, doc.content.strip())
            # What: Skip if already seen.
            # Why: Avoid duplicate context entries.
            if key in seen:
                continue

            # What: Mark fingerprint as seen.
            # Why: Future duplicates will be filtered.
            seen.add(key)
            # What: Keep this first occurrence.
            # Why: Preserve one representative doc.
            unique_docs.append(doc)

        # What: Return deduplicated docs.
        # Why: Downstream quality checks/generation should use clean context.
        return unique_docs

    # What: Determine whether retrieval relevance is too low.
    # Why: Low-relevance retrieval should trigger fallback path.
    @staticmethod
    def _has_low_similarity(documents: list[EmbeddedChunk]) -> bool:
        # What: Collected numeric similarity scores.
        # Why: Used to evaluate best-match quality.
        scores: list[float] = []

        # What: Walk through docs and collect available score metadata.
        # Why: Not every retriever exposes scores in identical keys.
        for doc in documents:
            # What: Safe metadata retrieval with empty fallback.
            # Why: Prevent None access errors.
            metadata = doc.metadata or {}
            # What: Read score from known keys.
            # Why: Support multiple retriever conventions.
            score = metadata.get("score", metadata.get("similarity"))
            # What: Keep only numeric values.
            # Why: Ignore malformed metadata.
            if isinstance(score, (int, float)):
                scores.append(float(score))

        # If the retriever does not expose scores, do not block on this check.
        # What: Skip score gate when unavailable.
        # Why: Avoid false negatives when retriever omits score metadata.
        if not scores:
            return False

        # What: Compare best score to threshold.
        # Why: Best doc should still be relevant enough.
        return max(scores) < RAGAgent.MIN_SIMILARITY_SCORE

    # What: Call SageMaker LLM through inference executor.
    # Why: Generate final answer from query + retrieved context.
    @staticmethod
    def _call_llm(query: str, context: str) -> str:
        # What: Local callable passed into circuit breaker.
        # Why: Circuit breaker controls and records each dependency attempt.
        def _call() -> str:
            # What: Construct endpoint client.
            # Why: Required to invoke tuned model in SageMaker.
            llm = LLMInferenceSagemakerEndpoint(
                endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
                inference_component_name=None,
            )
            # What: Execute inference with query/context.
            # Why: Produces model-generated answer text.
            return InferenceExecutor(llm=llm, query=query, context=context).execute()

        # What: Execute model call under circuit breaker.
        # Why: Opens circuit after repeated failures.
        return RAGAgent._sagemaker_cb.call(_call)

    # What: Main RAG method returning answer plus execution diagnostics.
    # Why: Supervisor/validator need metadata, not only final text.
    @staticmethod
    def invoke_with_details(query: str, allow_web_fallback: bool = True) -> dict[str, str | bool]:
        # What: Create retriever instance.
        # Why: Retrieves candidate chunks for grounding.
        retriever = ContextRetriever(mock=False)

        # What: Run retrieval behind circuit breaker.
        # Why: Fail fast when dependency is unhealthy.
        try:
            # What: Search using query with top-k=3.
            # Why: Small k keeps retrieval focused and low latency.
            retrieved_docs = RAGAgent._retriever_cb.call(retriever.search, query, 3)
        # What: Handle open retriever circuit.
        # Why: Avoid repeated failing calls and apply fallback policy.
        except CircuitBreakerOpenError:
            # What: Optional fallback to web.
            # Why: Keep service responsive when vector retrieval is down.
            if allow_web_fallback:
                return {
                    "answer": WebAgent.invoke(query),
                    "retrieved_context": "",
                    "used_web_fallback": True,
                }
            # What: Return explicit RAG-unavailable response.
            # Why: Caller requested no web fallback.
            return {
                "answer": "RAG retrieval is temporarily unavailable. Please try again.",
                "retrieved_context": "",
                "used_web_fallback": False,
            }
        # What: Catch any retriever runtime errors.
        # Why: Keep behavior deterministic under unexpected failures.
        except Exception:
            if allow_web_fallback:
                return {
                    "answer": WebAgent.invoke(query),
                    "retrieved_context": "",
                    "used_web_fallback": True,
                }
            return {
                "answer": "RAG retrieval failed. Please try again.",
                "retrieved_context": "",
                "used_web_fallback": False,
            }

        # What: Handle empty retrieval.
        # Why: No evidence means RAG cannot answer groundedly.
        if not retrieved_docs:
            if not allow_web_fallback:
                return {
                    "answer": "No relevant documents were found in the knowledge base.",
                    "retrieved_context": "",
                    "used_web_fallback": False,
                }
            return {
                "answer": WebAgent.invoke(query),
                "retrieved_context": "",
                "used_web_fallback": True,
            }

        # What: Remove duplicate chunks.
        # Why: Improve context quality and avoid repetition.
        deduplicated_docs = RAGAgent._deduplicate_docs(retrieved_docs)

        # What: Enforce minimum evidence count.
        # Why: Very small evidence set is low-confidence.
        if len(deduplicated_docs) < RAGAgent.MIN_DOCS:
            if not allow_web_fallback:
                return {
                    "answer": "Not enough relevant documents were found in the knowledge base.",
                    "retrieved_context": EmbeddedChunk.to_context(deduplicated_docs),
                    "used_web_fallback": False,
                }
            return {
                "answer": WebAgent.invoke(query),
                "retrieved_context": EmbeddedChunk.to_context(deduplicated_docs),
                "used_web_fallback": True,
            }

        # What: Reject low-relevance retrieval.
        # Why: Prevent hallucinations from weak grounding.
        if RAGAgent._has_low_similarity(deduplicated_docs):
            if not allow_web_fallback:
                return {
                    "answer": "Retrieved documents have low relevance for this query.",
                    "retrieved_context": EmbeddedChunk.to_context(deduplicated_docs),
                    "used_web_fallback": False,
                }
            return {
                "answer": WebAgent.invoke(query),
                "retrieved_context": EmbeddedChunk.to_context(deduplicated_docs),
                "used_web_fallback": True,
            }

        # What: Convert retrieved chunks into prompt context text.
        # Why: Inference executor expects context string.
        context = EmbeddedChunk.to_context(deduplicated_docs)
        # What: Attempt generation with circuit-breaker-protected LLM call.
        # Why: Convert grounded context into final answer.
        try:
            answer = RAGAgent._call_llm(query, context)
        # What: Handle open model circuit.
        # Why: Apply policy without hammering a failing endpoint.
        except CircuitBreakerOpenError:
            if allow_web_fallback:
                return {
                    "answer": WebAgent.invoke(query),
                    "retrieved_context": context,
                    "used_web_fallback": True,
                }
            return {
                "answer": "The generation service is temporarily unavailable.",
                "retrieved_context": context,
                "used_web_fallback": False,
            }
        # What: Handle model call errors.
        # Why: Keep robust fallback behavior.
        except Exception:
            if allow_web_fallback:
                return {
                    "answer": WebAgent.invoke(query),
                    "retrieved_context": context,
                    "used_web_fallback": True,
                }
            return {
                "answer": "The generation service failed.",
                "retrieved_context": context,
                "used_web_fallback": False,
            }

        # What: Return successful grounded result.
        # Why: Caller can inspect details for validation/telemetry.
        return {
            "answer": answer,
            "retrieved_context": context,
            "used_web_fallback": False,
        }

    # What: Simple public convenience API returning only answer text.
    # Why: Some callers only need string output, not diagnostic payload.
    @staticmethod
    def invoke(query: str) -> str:
        # What: Reuse detailed execution path.
        # Why: Avoid logic duplication.
        result = RAGAgent.invoke_with_details(query)
        # What: Return normalized string answer.
        # Why: Maintain stable return contract.
        return str(result["answer"])