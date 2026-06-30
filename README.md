# Procure-AI — Agentic RAG for IT Procurement

> An AI-powered IT product advisor that understands natural language requirements, filters by budget, compares specifications, and explains its recommendations — built with LangGraph, ChromaDB, FastAPI, and Gemini.

---

## Demo

Ask in natural language:
- *"Recommend a server for ML workloads under 200,000 THB"*
- *"Compare Dell vs HP laptops under 50,000 THB"*
- *"What network switches do you have under 30,000 THB?"*

The agent rewrites your query, retrieves relevant products via hybrid search, reranks results, applies budget filtering, and generates a recommendation with reasoning and source citations.

---

## Architecture

```
User Query
    ↓
FastAPI Gateway
    ↓
LangGraph Orchestrator (plan → act → reflect)
    ├── Query Rewriter      — expands ambiguous queries
    ├── Hybrid Search       — vector + BM25 keyword search
    ├── CrossEncoder Rerank — precision scoring
    ├── Budget Filter       — removes out-of-budget products
    ├── Spec Comparison     — side-by-side table for compare queries
    └── LLM Generate        — Gemini 2.0 Flash with source citations
    ↓
Streamlit Chat UI
```

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| LLM | Gemini 2.0 Flash | Fast, cost-effective, multilingual |
| Embedding | `all-MiniLM-L6-v2` | Local, no API key needed |
| Vector DB | ChromaDB | Local-first, zero infra overhead |
| Agent | LangGraph | Multi-step orchestration with state |
| RAG | LangChain | Mature ecosystem |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranking, free |
| Backend | FastAPI | Production-ready, async |
| Frontend | Streamlit | Fast to ship, no build step |
| Container | Docker + docker-compose | One-command reproducible env |

---

## Project Structure

```
procure-ai/
├── agent/
│   ├── graph.py          # LangGraph agent definition
│   ├── tools.py          # budget filter, query rewriter, spec compare
│   └── prompts.py        # system prompts
├── api/
│   └── main.py           # FastAPI endpoints
├── ingestion/
│   └── pipeline.py       # CSV → embed → ChromaDB
├── retrieval/
│   ├── hybrid_search.py  # vector + BM25
│   └── reranker.py       # CrossEncoder reranking
├── ui/
│   └── app.py            # Streamlit chat interface
├── eval/
│   └── ragas_eval.py     # RAGAS evaluation script
├── data/
│   └── sample_catalog.csv
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### With Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/{your-username}/procure-ai.git
cd procure-ai

# 2. Setup environment
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# 3. Run
docker compose up --build

# 4. Open
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

### Manual Setup

```bash
# 1. Create virtual environment
uv venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
uv pip install -r requirements.txt
uv pip install -e .

# 3. Configure environment
cp .env.example .env
# Add GOOGLE_API_KEY to .env

# 4. Ingest product catalog
python ingestion/pipeline.py

# 5. Run API
uvicorn api.main:app --reload

# 6. Run UI (separate terminal)
streamlit run ui/app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | Send query, receive answer + sources |
| `POST` | `/ingest` | Upload new CSV catalog |
| `GET` | `/products` | List all indexed products |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "recommend a server for ML workload under 200000 THB"}'
```

### Example Response

```json
{
  "answer": "Based on your requirements, I recommend the Dell PowerEdge R450 at 145,000 THB. This mid-range rack server features an Intel Xeon Silver 4314 CPU, 64GB RAM, and is specifically designed for ML inference workloads...",
  "sources": [
    {
      "product_id": "SRV002",
      "name": "Dell PowerEdge R450",
      "price_thb": 145000,
      "category": "Server"
    }
  ]
}
```

---

## Environment Variables

```bash
# .env.example

# LLM (Google Gemini)
LLM_API_KEY=your-google-api-key-here

# Embedding model (local, no key needed)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector database
CHROMA_PERSIST_DIR=./data/chroma

# App
APP_ENV=development
LOG_LEVEL=INFO
```

---

## Product Catalog Format

The system accepts CSV files with the following schema:

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Unique identifier |
| `name` | string | Product name |
| `category` | string | e.g. Laptop, Server, Network Switch |
| `brand` | string | e.g. Dell, HP, Cisco |
| `price_thb` | float | Price in Thai Baht |
| `specs` | JSON string | Technical specifications |
| `description` | string | Product description |

---

## Evaluation

Run RAGAS evaluation to measure retrieval and generation quality:

```bash
python eval/ragas_eval.py
```

Target metrics for MVP:

| Metric | Target |
|---|---|
| Faithfulness | > 0.75 |
| Answer Relevancy | > 0.70 |
| Budget Filter Accuracy | 100% |

---

## Roadmap

- [ ] Redis semantic cache to reduce LLM costs
- [ ] LangSmith tracing for observability
- [ ] Cloud deployment (GCP / AWS)
- [ ] Thai language support
- [ ] Real-time inventory sync
- [ ] User authentication

---

## Why This Project

IT distributors spend significant staff time manually answering product inquiries. This system demonstrates how Agentic RAG can automate that workflow — combining retrieval precision (hybrid search + reranking) with business logic (budget filtering, spec comparison) and natural language generation.

Built as a portfolio project targeting AI Engineer roles focused on GenAI adoption and end-to-end deployment.
