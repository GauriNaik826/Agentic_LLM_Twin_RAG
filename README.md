# Agentic LLM Twin RAG

Production-style LLM system that builds a personal knowledge twin from web content, indexes it for retrieval, generates fine-tuning datasets, and supports training/evaluation/inference pipelines using ZenML.

---

## 🚀 What this project does

This repository implements an end-to-end LLM workflow:

1. **Data ETL**: Crawls source links and stores raw documents.
2. **Feature Engineering**: Cleans, chunks, embeds, and pushes content to vector storage.
3. **Dataset Generation**: Creates instruction and preference datasets.
4. **Training**: Runs model fine-tuning workflows.
5. **Evaluation**: Evaluates model quality.
6. **Inference**: Exposes a FastAPI service for RAG/inference use cases.

---

## 🧱 Project structure

```text
.
├── configs/                # ZenML YAML configs (settings + parameters)
├── llm_engineering/        # Core package (domain, application, model, infrastructure)
├── pipelines/              # ZenML pipeline definitions
├── steps/                  # ZenML step implementations used by pipelines
├── tools/                  # CLI entry points and utility scripts
├── tests/                  # Unit and integration tests
├── data/                   # Local artifacts and sample/raw data
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

### Key folders

- **`llm_engineering/domain`**: Business entities/types (`Document`, `Chunk`, prompts, queries, etc.).
- **`llm_engineering/application`**: Pipeline-facing business logic (crawling, preprocessing, RAG flows).
- **`llm_engineering/model`**: Training, inference, and evaluation model logic.
- **`llm_engineering/infrastructure`**: External integrations (DB, AWS, API, I/O).

---

## ⚙️ Requirements

- Python **3.11**
- Poetry **>=1.8,<2.0**
- Docker + Docker Compose
- (Optional) AWS CLI for cloud workflows

---

## 🛠️ Setup

```bash
git clone <your-repo-url>
cd Agentic_LLM_Twin_RAG

poetry env use 3.11
poetry install --without aws
poetry run pre-commit install
```

Create env file:

```bash
cp .env.example .env
```

At minimum, set required secrets in `.env` (for example OpenAI key) before running pipelines.

---

## ▶️ How to run

> Main CLI entrypoint is `tools/run.py` and should be called as module.

### 1) Bring up local infra

```bash
docker compose up -d
poetry run zenml logout --local
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES poetry run zenml login --local
```

### 2) Run ETL

```bash
poetry run python -m tools.run --run-etl --etl-config-filename digital_data_etl_paul_iusztin.yaml
```

### 3) Run feature engineering

```bash
poetry run python -m tools.run --run-feature-engineering
```

### 4) Generate datasets

```bash
poetry run python -m tools.run --run-generate-instruct-datasets
poetry run python -m tools.run --run-generate-preference-datasets
```

### 5) Train and evaluate

```bash
poetry run python -m tools.run --run-training
poetry run python -m tools.run --run-evaluation
```

### 6) Serve inference API

```bash
poetry run uvicorn tools.ml_service:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🧩 How configs work

Each file in `configs/` has:

- `settings`: execution/runtime options (docker parent image, orchestrator behavior).
- `parameters`: arguments injected into the matching ZenML pipeline function.

`tools/run.py` selects a YAML based on your CLI flags and passes it through `with_options(config_path=...)`.

Example mapping:

- `configs/digital_data_etl_maxime_labonne.yaml` → `pipelines/digital_data_etl.py`
- `configs/feature_engineering.yaml` → `pipelines/feature_engineering.py`
- `configs/training.yaml` → `pipelines/training.py`

---

## 🧪 Quality checks

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest tests/
```

Fix formatting/lint automatically:

```bash
poetry run ruff check --fix .
poetry run ruff format .
```

---

## 🐳 Docker

Build local image:

```bash
docker buildx build --platform linux/amd64 -t llmtwin -f Dockerfile .
```

Run end-to-end data pipeline in container:

```bash
docker run --rm --network host --shm-size=2g --env-file .env llmtwin poetry run python -m tools.run --no-cache --run-end-to-end-data
```

---

## ❗Common issues

1. **Command not found / wrong entrypoint**  
   Use `python -m tools.run ...` instead of `python run.py`.

2. **Pipeline fails due to missing secrets**  
   Ensure `.env` contains required credentials (OpenAI, optional service keys).

3. **Infra connection errors (Mongo/Qdrant/ZenML)**  
   Start local infra first with Docker + local ZenML login.

4. **Long-running ETL jobs**  
   Reduce links in your ETL config YAML for quick smoke tests.

---

## 📌 Useful commands (quick reference)

```bash
# ETL (Paul or Maxime config)
poetry run python -m tools.run --run-etl --etl-config-filename digital_data_etl_paul_iusztin.yaml
poetry run python -m tools.run --run-etl --etl-config-filename digital_data_etl_maxime_labonne.yaml

# End-to-end data flow
poetry run python -m tools.run --run-end-to-end-data

# Export artifacts to JSON
poetry run python -m tools.run --run-export-artifact-to-json
```

---

## 📄 License

MIT (see `LICENSE`).

---

## 🙌 Acknowledgments

This project builds upon concepts from the LLM Engineers Handbook and extends it with an agent-driven RAG inference system and production-oriented deployment improvements.
