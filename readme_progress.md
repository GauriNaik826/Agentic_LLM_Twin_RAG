
# 🚀 Agentic LLM Twin RAG – Data Engineering Phase

## 📌 Project Overview

Agentic LLM Twin is an end-to-end LLM engineering system for building a personalized AI model from curated digital content. It covers the full ML lifecycle: data ingestion, feature engineering, fine-tuning, retrieval-augmented generation (RAG), evaluation, and deployment. Beyond a traditional LLM Twin, it adds an agentic retrieval layer that plans, decomposes, and executes multi-step information gathering before response generation. The result is a production-grade, personalized writing co-pilot built on modular pipelines and cloud-ready infrastructure.

The system is designed following modern ML engineering and MLOps best practices.

---

# 🏗 Architecture (Current Phase)

The current implementation covers the **Data Engineering and Orchestration layer**.

```
Web Sources → ETL Pipeline → MongoDB (Raw Data)
                           → ZenML (Artifact Tracking)
```

Core components:

* `llm_engineering/` → Domain & application logic
* `pipelines/` → ZenML pipeline definitions
* `steps/` → Modular step-level logic
* `configs/` → YAML runtime configuration
* `tools/run.py` → CLI entrypoint
* `docker-compose.yml` → Local infrastructure

---

# ⚙️ Tech Stack

| Layer               | Tool         | Purpose                                       |
| ------------------- | ------------ | --------------------------------------------- |
| Python Version      | pyenv        | Deterministic interpreter control             |
| Dependency Mgmt     | Poetry       | Reproducible environments                     |
| Orchestration       | ZenML        | ML pipeline orchestration & artifact tracking |
| NoSQL DB            | MongoDB      | Raw unstructured data storage                 |
| Vector DB           | Qdrant       | (Next phase) Embedding storage                |
| Containerization    | Docker       | Local infrastructure                          |
| Model Registry      | Hugging Face | Future model versioning                       |
| Experiment Tracking | Comet ML     | Future training monitoring                    |
| Prompt Monitoring   | Opik         | Future LLM trace monitoring                   |

---

# 🧠 What Has Been Implemented

## ✅ 1. Environment Setup

* Python 3.11.8 pinned using `pyenv`
* Dependencies installed via Poetry
* Deterministic dependency locking through `poetry.lock`

```bash
pyenv install 3.11.8
pyenv local 3.11.8
poetry install
```

---

## ✅ 2. Docker-Based Local Infrastructure

MongoDB and Qdrant are provisioned locally using Docker.

```bash
docker compose up -d
docker ps
```

Running services:

* MongoDB → `localhost:27017`
* Qdrant → `localhost:6333`

---

## ✅ 3. ZenML Pipeline Orchestration

Implemented a reproducible ETL pipeline using ZenML:

```python
@pipeline
def digital_data_etl(user_full_name: str, links: list[str]):
    user = get_or_create_user(user_full_name)
    crawl_links(user=user, links=links)
```

ZenML handles:

* Step orchestration
* DAG visualization
* Artifact storage
* Metadata tracking
* Run versioning

Pipeline execution:

```bash
poetry run python tools/run.py --run-etl
```

---

## ✅ 4. Config-Driven Execution (YAML)

Runtime behavior is controlled via YAML files in `configs/`.

Example:

```yaml
parameters:
  user_full_name: Gauri Naik
  links:
    - https://yourblog.com/post1
```

This enables:

* Reusable pipelines
* No code changes for new datasets
* Clean runtime parameter injection

---

## ✅ 5. Artifact & Metadata Tracking

Each ZenML step output is stored as a versioned artifact.

Metadata includes:

* Crawled domains
* Number of links
* Success/failure counts
* Dataset statistics (future phase)

This ensures reproducibility and observability.

---

## ✅ 6. Hugging Face Authentication

Configured model registry access:

```bash
huggingface-cli login
```

This prepares the system for:

* Model downloads
* Fine-tuned model uploads
* Versioned model sharing

---

# 📊 ZenML Dashboard

View pipeline runs and artifacts:

```
http://127.0.0.1:8237
```

This shows:

* DAG execution graph
* Step logs
* Artifacts
* Metadata
* Stack configuration

---

# 🔜 Next Implementation Phase

Upcoming components:

* Feature engineering (chunking + embedding generation)
* Vector storage integration with Qdrant
* Instruction dataset generation
* Preference dataset (DPO)
* LLM fine-tuning
* Agentic RAG inference layer
* Cloud deployment via AWS SageMaker

---

# 🎯 Engineering Principles Applied

* Separation of orchestration and business logic
* Modular step-based design
* Config-driven pipelines
* Infrastructure abstraction
* Artifact versioning
* Reproducible ML workflows
* Cloud-ready architecture

---

# 🚦 How to Run

1️⃣ Install dependencies

```bash
poetry install
```

2️⃣ Start infrastructure

```bash
docker compose up -d
```

3️⃣ Run ETL pipeline

```bash
poetry run python tools/run.py --run-etl
```

---

# 📌 Current Status

🟢 Data Engineering Phase Complete
🟡 Feature Engineering Phase Next
🔵 Agentic RAG & Training Phase Upcoming

---

If you'd like, I can now:

* Make a more concise resume-optimized README
* Add an architecture diagram section
* Add an “Agentic Extension Roadmap” section
* Or tailor this specifically for ML SWE applications

Just tell me the direction you want.
