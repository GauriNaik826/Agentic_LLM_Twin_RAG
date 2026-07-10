# What: Postpone evaluation of type annotations.
# Why: Lets us use forward references and modern typing patterns safely.
from __future__ import annotations

# What: Import typing helpers for state schema and route constraints.
# Why: Improves type safety and self-documentation for graph state data.
from typing import Any, Literal, TypedDict
import re

# What: Import LangGraph core primitives.
# Why: We need START/END sentinels and StateGraph builder to define workflow orchestration.
from langgraph.graph import END, START, StateGraph

# What: Import specialized worker agents.
# Why: Supervisor routes queries to one of these execution agents.
from llm_engineering.application.agents.rag_agent import RAGAgent
from llm_engineering.application.agents.twin_writer import TwinWriterAgent
from llm_engineering.application.agents.web_agent import WebAgent, WebSearchTimeoutError
from llm_engineering.application.guardrails import (
    CircuitBreakerOpenError,
    InputGuardrail,
    OutputValidator,
    UnsafePromptException,
    UnsupportedRequestException,
)
# What: Import router that classifies query intent.
# Why: Route selection should be isolated from execution logic.
from llm_engineering.application.orchestration.router import QueryRouter
# What: Import shared in-memory state store.
# Why: Maintains conversation and run metadata across invocations.
from llm_engineering.application.orchestration.state import shared_supervisor_state


# What: Define the LangGraph state contract.
# Why: Every node reads/writes this shared structure.
class SupervisorGraphState(TypedDict, total=False):
    # What: Original user query.
    # Why: Required by router and all agent nodes.
    query: str
    # What: Selected route name.
    # Why: Used by conditional edges and traceability.
    route: str
    # What: Running chat history.
    # Why: Enables future context-aware routing and responses.
    conversation_history: list[dict[str, str]]
    # What: Retrieved RAG context cache.
    # Why: Placeholder for future validator/citation checks.
    retrieved_context: str
    # What: Raw/processed web results.
    # Why: Preserves fetched evidence for downstream validators.
    web_results: list[dict[str, Any]]
    # What: Final answer string.
    # Why: Output payload returned to API caller.
    answer: str
    # What: Free-form metadata map.
    # Why: Stores routing/execution diagnostics without rigid schema migration.
    metadata: dict[str, Any]
    # What: Failure counter.
    # Why: Supports future retry/circuit-breaker logic.
    failure_count: int


# What: Supervisor orchestration entrypoint.
# Why: Centralizes routing, execution, validation, and state synchronization.
class Supervisor:
    # What: Simple URL detector for citation checks.
    # Why: Lets validator confirm citation presence for web/rag answers.
    _URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

    # What: Pattern list for potentially toxic/unsafe content.
    # Why: Enforces output safety before returning final answer.
    _TOXICITY_PATTERNS = [
        re.compile(r"\bkill\b", re.IGNORECASE),
        re.compile(r"\bterror(?:ist|ism)?\b", re.IGNORECASE),
        re.compile(r"\bhate\s+\w+", re.IGNORECASE),
        re.compile(r"\bmake\s+(a\s+)?bomb\b", re.IGNORECASE),
        re.compile(r"\bself-harm\b", re.IGNORECASE),
    ]

    # What: Unsupported instruction phrases for hallucination-style safety checks.
    # Why: If these appear in output, confidence should be lowered.
    _UNSAFE_ADVICE_PATTERNS = [
        re.compile(r"\bhack\b", re.IGNORECASE),
        re.compile(r"\bphish(?:ing)?\b", re.IGNORECASE),
        re.compile(r"\bsteal\b", re.IGNORECASE),
        re.compile(r"\bmalware\b", re.IGNORECASE),
    ]

    # What: Minimum and maximum output length bounds.
    # Why: Detect answers that are too short to be useful or too long/noisy.
    MIN_WORDS = 20
    MAX_WORDS = 500

    # What: Initialize router and compiled graph once.
    # Why: Avoid rebuilding expensive graph structure on every request.
    def __init__(self) -> None:
        self._input_guardrail = InputGuardrail()
        self._output_validator = OutputValidator()
        # What: Create route classifier.
        # Why: Supervisor node delegates intent detection to this component.
        self._router = QueryRouter()
        # What: Compile graph at startup.
        # Why: Reuse compiled pipeline for lower per-request overhead.
        self._graph = self._build_graph()

    def _input_guardrail_node(self, state: SupervisorGraphState) -> SupervisorGraphState:
        query = state["query"]
        metadata = dict(state.get("metadata", {}))

        cleaned_query, guardrail_metadata = self._input_guardrail.process(query)
        metadata.update(guardrail_metadata)

        return {
            "query": cleaned_query,
            "metadata": metadata,
        }

    # What: Supervisor node that decides route.
    # Why: Keeps route decision as explicit first graph step.
    def _supervisor_node(self, state: SupervisorGraphState) -> SupervisorGraphState:
        # What: Read query from state.
        # Why: Router needs the user input to classify intent.
        query = state["query"]
        # What: Classify query into route.
        # Why: Determines which specialized agent node should run next.
        route = self._router.classify(query)

        # What: Copy metadata map.
        # Why: Avoid mutating shared dictionary references from prior state.
        metadata = dict(state.get("metadata", {}))
        # What: Record selected route.
        # Why: Useful for observability/debugging and post-run analysis.
        metadata["selected_route"] = route

        # What: Return state delta for downstream nodes.
        # Why: LangGraph merges this output into global graph state.
        return {
            "route": route,
            "metadata": metadata,
        }

    # What: Route selector used by conditional edge.
    # Why: Guarantees only known branches are returned.
    def _route_condition(self, state: SupervisorGraphState) -> Literal["rag", "web", "twin_writer"]:
        # What: Read proposed route with safe default.
        # Why: Prevent missing-key crashes and default to reliable path.
        route = state.get("route", "rag")
        # What: Validate route token.
        # Why: Protect graph from unexpected classifier outputs.
        if route not in {"rag", "web", "twin_writer"}:
            return "rag"

        # What: Return validated route.
        # Why: Drives conditional edge to the correct agent node.
        return route

    # What: RAG execution node.
    # Why: Handles queries requiring internal knowledge retrieval.
    def _rag_node(self, state: SupervisorGraphState) -> SupervisorGraphState:
        # What: Invoke RAG agent.
        # Why: Produce grounded answer from indexed content.
        rag_result = RAGAgent.invoke_with_details(state["query"])
        answer = str(rag_result["answer"])
        retrieved_context = str(rag_result.get("retrieved_context", ""))
        used_web_fallback = bool(rag_result.get("used_web_fallback", False))
        # What: Copy metadata map.
        # Why: Keep prior metadata and append execution details safely.
        metadata = dict(state.get("metadata", {}))
        # What: Tag executed agent.
        # Why: Track actual execution branch taken.
        metadata["executed_agent"] = "rag"
        metadata["rag_used_web_fallback"] = used_web_fallback
        # What: Return answer and updated metadata.
        # Why: Feed validator and final response extraction.
        return {
            "answer": answer,
            "retrieved_context": retrieved_context,
            "metadata": metadata,
        }

    # What: Web execution node.
    # Why: Handles fresh/time-sensitive questions via web retrieval.
    def _web_node(self, state: SupervisorGraphState) -> SupervisorGraphState:
        # What: Invoke web agent.
        # Why: Gather and summarize recent external information.
        query = state["query"]
        metadata = dict(state.get("metadata", {}))

        try:
            web_result = WebAgent.invoke_with_details(query)
            answer = str(web_result["answer"])
            web_results = list(web_result.get("web_results", []))
            metadata["web_provider"] = web_result.get("provider", "unknown")
            metadata["web_sources"] = web_result.get("sources", [])
            metadata["executed_agent"] = "web"
            return {
                "answer": answer,
                "web_results": web_results,
                "metadata": metadata,
            }
        except CircuitBreakerOpenError:
            # Phase 2 policy: open web circuit -> immediate fallback to RAG.
            rag_result = RAGAgent.invoke_with_details(query, allow_web_fallback=False)
            metadata["executed_agent"] = "rag"
            metadata["fallback_from"] = "web_circuit_open"
            return {
                "answer": str(rag_result["answer"]),
                "retrieved_context": str(rag_result.get("retrieved_context", "")),
                "metadata": metadata,
            }
        except WebSearchTimeoutError:
            # Phase 3 policy: web timeout -> Twin Writer.
            metadata["fallback_from"] = "web_timeout"
            answer = TwinWriterAgent.invoke(query)
            metadata["executed_agent"] = "twin_writer"
            return {
                "answer": answer,
                "metadata": metadata,
            }
        except Exception:
            metadata["fallback_from"] = "web_failure"
            answer = TwinWriterAgent.invoke(query)
            metadata["executed_agent"] = "twin_writer"
            return {
                "answer": answer,
                "metadata": metadata,
            }

    # What: Twin writer execution node.
    # Why: Handles style-centric generation without retrieval.
    def _twin_writer_node(self, state: SupervisorGraphState) -> SupervisorGraphState:
        # What: Invoke fine-tuned writer agent.
        # Why: Generate stylistic output directly from user prompt.
        try:
            answer = TwinWriterAgent.invoke(state["query"])
        except Exception:
            # Phase 3 policy: Twin writer hard failure -> safe response.
            answer = "I cannot generate that response right now. Please try a different request."
        # What: Copy metadata map.
        # Why: Preserve existing metadata while annotating branch info.
        metadata = dict(state.get("metadata", {}))
        # What: Tag executed agent.
        # Why: Makes branch behavior explicit in trace data.
        metadata["executed_agent"] = "twin_writer"
        # What: Return answer bundle.
        # Why: Pass output to validator/final state.
        return {"answer": answer, "metadata": metadata}

    # What: Validator node placeholder.
    # Why: Provides explicit post-processing hook for future checks.
    def _validator_node(self, state: SupervisorGraphState) -> SupervisorGraphState:
        metadata = dict(state.get("metadata", {}))
        answer = state.get("answer", "")
        route = state.get("route", metadata.get("selected_route", "rag"))
        retrieved_context = state.get("retrieved_context", "")
        web_results = state.get("web_results", [])
        query = state.get("query", "")

        validation_result = self._output_validator.validate(
            route=str(route),
            query=str(query),
            answer=str(answer),
            retrieved_context=str(retrieved_context),
            web_results=list(web_results),
        )

        metadata["validated"] = validation_result.passed
        metadata["validation_checks"] = validation_result.checks
        metadata["validation_reason"] = validation_result.reason
        metadata["confidence"] = validation_result.confidence

        if validation_result.checks.get("toxicity_ok") is False:
            safe_answer = "I cannot provide that response safely. Please rephrase your request."
            return {"answer": safe_answer, "metadata": metadata}

        return {"metadata": metadata}

    # What: Construct and compile LangGraph workflow.
    # Why: Defines deterministic control flow between supervisor and agent nodes.
    def _build_graph(self):
        # What: Create graph builder for our typed state.
        # Why: Enforces node I/O shape through shared state schema.
        graph_builder = StateGraph(SupervisorGraphState)

        # What: Register supervisor node.
        # Why: This node computes the route before execution nodes.
        graph_builder.add_node("input_guardrail", self._input_guardrail_node)
        graph_builder.add_node("supervisor", self._supervisor_node)
        # What: Register RAG node.
        # Why: Route target for knowledge-grounded tasks.
        graph_builder.add_node("rag", self._rag_node)
        # What: Register web node.
        # Why: Route target for latest-news/time-sensitive tasks.
        graph_builder.add_node("web", self._web_node)
        # What: Register twin writer node.
        # Why: Route target for style-only generation tasks.
        graph_builder.add_node("twin_writer", self._twin_writer_node)
        # What: Register validator node.
        # Why: Shared final quality/guardrail checkpoint before END.
        graph_builder.add_node("validator", self._validator_node)

        # What: Link graph start to input guardrail.
        # Why: Every request must be sanitized and checked before routing.
        graph_builder.add_edge(START, "input_guardrail")
        graph_builder.add_edge("input_guardrail", "supervisor")
        # What: Add route-based branching edges.
        # Why: Dynamically choose execution node based on classifier output.
        graph_builder.add_conditional_edges(
            "supervisor",
            self._route_condition,
            {
                # What: Map rag route token to rag node.
                # Why: Ensures deterministic branch lookup.
                "rag": "rag",
                # What: Map web route token to web node.
                # Why: Enables web branch execution.
                "web": "web",
                # What: Map twin route token to twin_writer node.
                # Why: Enables style branch execution.
                "twin_writer": "twin_writer",
            },
        )
        # What: Route rag output into validator.
        # Why: Apply uniform post-processing regardless of branch.
        graph_builder.add_edge("rag", "validator")
        # What: Route web output into validator.
        # Why: Keep output handling consistent across agents.
        graph_builder.add_edge("web", "validator")
        # What: Route twin output into validator.
        # Why: Keep output handling consistent across agents.
        graph_builder.add_edge("twin_writer", "validator")
        # What: Mark validator as final processing step.
        # Why: End graph only after validation stage completes.
        graph_builder.add_edge("validator", END)

        # What: Compile graph for runtime invocation.
        # Why: Produces executable graph object used per request.
        return graph_builder.compile()

    # What: Public API to run supervisor workflow.
    # Why: Called by FastAPI to process each user query.
    def invoke(self, query: str) -> str:
        # What: Read persisted shared memory snapshot.
        # Why: Carry prior context and counters into this run.
        memory_state = shared_supervisor_state.get()
        # What: Append current user message to history.
        # Why: Preserve conversation continuity for future enhancements.
        conversation_history = memory_state.conversation_history + [{"role": "user", "content": query}]

        # What: Build graph input state.
        # Why: Provide router/agents with query plus current memory context.
        graph_input: SupervisorGraphState = {
            "query": query,
            "conversation_history": conversation_history,
            "retrieved_context": memory_state.retrieved_context,
            "web_results": memory_state.web_results,
            "answer": memory_state.answer,
            "metadata": dict(memory_state.metadata),
            "failure_count": memory_state.failure_count,
        }

        # What: Execute LangGraph workflow.
        # Why: Runs guardrails, route selection, and selected agent branch end-to-end.
        try:
            result = self._graph.invoke(graph_input)
        except (UnsafePromptException, UnsupportedRequestException) as exc:
            failure_count = memory_state.failure_count + 1
            metadata = dict(memory_state.metadata)
            metadata.update(
                {
                    "guardrail_blocked": True,
                    "guardrail_reason": exc.__class__.__name__,
                }
            )
            shared_supervisor_state.update(
                query=query,
                route="blocked",
                conversation_history=conversation_history,
                answer=str(exc),
                metadata=metadata,
                failure_count=failure_count,
            )
            return str(exc)

        # What: Extract final answer.
        # Why: Return plain string to API response contract.
        answer = result.get("answer", "")

        # Supervisor recovery policy when validator fails.
        result_metadata = dict(result.get("metadata", {}))
        if not result_metadata.get("validated", True):
            route = str(result.get("route", "rag"))
            reason = str(result_metadata.get("validation_reason", "validation_failed"))

            if route == "rag":
                # Validation failed on RAG -> fallback to Web.
                try:
                    web_result = WebAgent.invoke_with_details(query)
                    answer = str(web_result["answer"])
                    result["web_results"] = list(web_result.get("web_results", []))
                    result_metadata["recovery_action"] = "fallback_rag_to_web"
                except Exception:
                    answer = "I could not validate the grounded response. Please try rephrasing your request."
                    result_metadata["recovery_action"] = "fallback_rag_to_safe"
            elif route == "web":
                # Validation failed on Web -> fallback to Twin Writer.
                try:
                    answer = TwinWriterAgent.invoke(query)
                    result_metadata["recovery_action"] = "fallback_web_to_twin"
                except Exception:
                    answer = "I could not validate the web response. Please try again later."
                    result_metadata["recovery_action"] = "fallback_web_to_safe"
            else:
                # Twin validation failed -> safe response.
                answer = "I could not generate a sufficiently reliable style-matched answer."
                result_metadata["recovery_action"] = "fallback_twin_to_safe"

            result_metadata["recovery_reason"] = reason
            result["metadata"] = result_metadata

        # What: Persist updated state back into shared memory.
        # Why: Keep route/result metadata and history available for next turn.
        shared_supervisor_state.update(
            query=query,
            route=result.get("route", "rag"),
            conversation_history=result.get("conversation_history", conversation_history),
            retrieved_context=result.get("retrieved_context", memory_state.retrieved_context),
            web_results=result.get("web_results", memory_state.web_results),
            answer=answer,
            metadata=result.get("metadata", memory_state.metadata),
            failure_count=result.get("failure_count", memory_state.failure_count),
        )

        # What: Return generated answer.
        # Why: Final output consumed by FastAPI endpoint and client.
        return answer