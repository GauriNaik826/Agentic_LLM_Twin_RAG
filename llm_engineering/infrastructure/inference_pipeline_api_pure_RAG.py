import opik
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.orchestration.supervisor import Supervisor
from llm_engineering.application.orchestration.state import shared_supervisor_state
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()
supervisor = Supervisor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


class QueryDetailsResponse(BaseModel):
    answer: str
    agent_used: str
    sources: list[str]
    grounded: bool
    confidence: float
    latency_sec: float
    metadata: dict


def _build_dashboard_response(query: str) -> dict:
    start = perf_counter()
    answer = supervisor.invoke(query)
    latency_sec = round(perf_counter() - start, 3)

    state = shared_supervisor_state.get()
    metadata = state.metadata or {}
    validation_checks = metadata.get("validation_checks", {})

    route = state.route or metadata.get("selected_route", "rag")
    executed_agent = metadata.get("executed_agent", route)
    agent_map = {
        "rag": "Advanced RAG",
        "web": "Web Agent",
        "twin_writer": "Twin Writer",
    }
    agent_used = agent_map.get(executed_agent, "Advanced RAG")

    sources = []
    web_sources = metadata.get("web_sources") or []
    for source in web_sources:
        if isinstance(source, str) and source.strip():
            sources.append(source.strip())

    grounded = bool(validation_checks.get("grounding_ok", metadata.get("validated", False)))
    confidence = float(metadata.get("confidence", 0.0))

    return {
        "answer": answer,
        "agent_used": agent_used,
        "sources": sources,
        "grounded": grounded,
        "confidence": confidence,
        "latency_sec": latency_sec,
        "metadata": metadata,
    }


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        response = _build_dashboard_response(request.query)
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rag/details", response_model=QueryDetailsResponse)
async def rag_details_endpoint(request: QueryRequest):
    try:
        return _build_dashboard_response(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e