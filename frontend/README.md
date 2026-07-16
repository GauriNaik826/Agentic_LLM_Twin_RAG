# Agentic LLM Twin Dashboard (React)

A lightweight React + Vite dashboard that visualizes your FastAPI `/rag` endpoint.

## Prerequisites

- Node.js 18+
- npm

## Setup

```bash
cd frontend
npm install
cp .env.example .env
```

## Run

```bash
npm run dev
```

App URL: `http://localhost:5173`

## Backend

Run FastAPI from the project root:

```bash
python tools/ml_service.py
```

API URL used by frontend is configured via `VITE_API_BASE_URL`.
