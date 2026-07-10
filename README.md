# 🤖 Agentic LLM Twin – Personalized AI Writing Assistant

An end-to-end **LLM Engineering** project that builds a personalized AI assistant from an author's digital footprint.

The system combines:

- 📚 Advanced Retrieval-Augmented Generation (RAG)
- 🧠 Fine-tuned Llama models (SFT + DPO)
- 🤖 LangGraph Supervisor-based Multi-Agent Orchestration
- 🛡️ AI Guardrails & Circuit Breakers
- ☁️ AWS SageMaker Deployment
- ⚙️ End-to-End LLMOps with ZenML

Unlike traditional RAG systems that always retrieve documents before generation, this project introduces an **agentic inference layer** where a Supervisor dynamically selects the best strategy for each query.

---

# 🚀 Features

## ✅ End-to-End LLM Pipeline

The project covers the complete LLM lifecycle:

- Data ingestion
- Feature engineering
- Dataset generation
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Evaluation
- Deployment
- Agentic inference

---

# 🏗 System Architecture

## Offline Pipeline

```text
Digital Footprint
        │
        ▼
MongoDB
        │
        ▼
Cleaning
        │
        ▼
Chunking
        │
        ▼
Embedding
        │
        ▼
Qdrant Vector Store
        │
        ▼
Instruction Dataset
        │
        ▼
Preference Dataset
        │
        ▼
SFT
        │
        ▼
DPO
        │
        ▼
AWS SageMaker
```

---

## Online Agentic Inference

```text
                     User
                       │
                       ▼
                    FastAPI
                       │
                       ▼
              Input Guardrails
                       │
                       ▼
          LangGraph Supervisor
                       │
          Intent Classification
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 Advanced RAG     Web Search     Twin Writer
        │              │              │
        ▼              ▼              ▼
   SageMaker       Search API     SageMaker
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              Output Validator
                       │
                       ▼
              Final Response
```

---

# 🧠 Multi-Agent Design

Instead of sending every query through a fixed RAG pipeline, a **Supervisor Agent** first classifies the user's request and dynamically selects the most appropriate specialist.

### Advanced RAG Agent

Uses internal knowledge.

Capabilities:

- Query Expansion
- Metadata Extraction
- Filtered Semantic Search
- Cross-Encoder Reranking
- Retrieval Validation

Best suited for:

- Questions about the author's knowledge
- Internal documentation
- Personalized factual responses

---

### Web Search Agent

Uses external search APIs.

Capabilities:

- Live web search
- Content extraction
- Evidence summarization

Best suited for:

- Recent news
- Time-sensitive questions
- External information

---

### Twin Writer Agent

Uses the fine-tuned Llama model deployed on SageMaker.

Capabilities:

- Personalized writing
- Style transfer
- Content generation

Best suited for:

- LinkedIn posts
- Blogs
- Emails
- Writing assistance

---

# 🛡 AI Guardrails

The inference pipeline is protected through multiple layers of guardrails.

### Input Guardrails

Executed **before** routing.

Features:

- Prompt Injection Detection
- PII Masking
- Unsupported Request Detection
- Query Normalization

---

### Tool Guardrails

Executed inside each specialist agent.

Examples:

- Empty Retrieval Detection
- Duplicate Removal
- Similarity Threshold Validation
- Retrieval Quality Validation

---

### Output Guardrails

Executed before returning the response.

Checks include:

- Grounding
- Citation Validation
- Style Validation
- Toxicity Detection
- Hallucination Signal
- Confidence Estimation

---

# ⚡ Fault Tolerance

The system implements the **Circuit Breaker Pattern** to improve reliability.

Protected services:

- Retriever
- Web Search
- AWS SageMaker Endpoint

State Machine:

```text
Closed
   │
Repeated Failures
   ▼
Open
   │
Recovery Timeout
   ▼
Half Open
   │
Success / Failure
   ▼
Closed / Open
```

Fallback examples:

- Weak retrieval → Web Search
- Web timeout → Twin Writer
- Validation failure → Supervisor retries another strategy
- Multiple failures → Safe fallback response

---

# 📂 Project Structure

```text
.
├── configs/
├── data/
├── llm_engineering/
│
│   ├── application/
│   │   ├── agents/
│   │   ├── guardrails/
│   │   ├── orchestration/
│   │   ├── rag/
│   │   └── preprocessing/
│   │
│   ├── domain/
│   ├── infrastructure/
│   ├── model/
│   └── settings.py
│
├── pipelines/
├── steps/
├── tests/
├── tools/
└── pyproject.toml
```

---

# ⚙ Technology Stack

## AI

- LangGraph
- LangChain
- Hugging Face Transformers
- Llama 3
- OpenAI
- PyTorch

---

## MLOps

- ZenML
- MLflow
- Docker
- AWS SageMaker

---

## Data

- MongoDB
- Qdrant
- PostgreSQL

---

## Backend

- FastAPI
- REST APIs

---

# 🚀 Setup

```bash
git clone <repo>
cd Agentic_LLM_Twin_RAG

poetry install

cp .env.example .env
```

---

# ▶ Run

Start infrastructure

```bash
docker compose up -d
```

Run ETL

```bash
poetry run python -m tools.run --run-etl
```

Run Feature Engineering

```bash
poetry run python -m tools.run --run-feature-engineering
```

Train

```bash
poetry run python -m tools.run --run-training
```

Evaluate

```bash
poetry run python -m tools.run --run-evaluation
```

Serve

```bash
poetry run uvicorn tools.ml_service:app --reload
```

---

# API

```http
POST /rag
```

Request

```json
{
  "query":"Write a LinkedIn post about RAG."
}
```

Response

```json
{
  "answer":"..."
}
```
---

# License

MIT License

---

# Acknowledgements

Built upon concepts from the **LLM Engineers Handbook**, extended with:

- Agentic AI
- LangGraph Orchestration
- Production Guardrails
- Circuit Breakers
- Supervisor-based Multi-Agent Systems

