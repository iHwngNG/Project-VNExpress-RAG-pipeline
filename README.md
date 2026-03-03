# 📰 Vietnamese News Assistant — RAG Pipeline

A question-answering system for Vietnamese news using **Retrieval-Augmented Generation (RAG)**.  
Automatically finds relevant articles from the database and answers based on real content — **no hallucination**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Ollama](https://img.shields.io/badge/LLM-Gemma3:4b-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
  - [Option 1: Docker Compose (Recommended)](#option-1-docker-compose-recommended)
  - [Option 2: Run Locally](#option-2-run-locally)
  - [Option 3: Google Colab](#option-3-google-colab)
- [Configuration](#-configuration)
- [RAG Pipeline Details](#-rag-pipeline-details)
- [GPU Support](#-gpu-support)
- [Troubleshooting](#-troubleshooting)

---

## 🔍 Overview

The user asks a question in Vietnamese → The system automatically:

1. **Analyzes the question** — generates 5 diverse queries from the prompt
2. **Searches** — retrieves relevant chunks from the database (ChromaDB)
3. **Filters** — keeps only articles matching the right topic/category
4. **Ranks** — selects the K most accurate articles (cross-encoder)
5. **Answers** — LLM synthesizes a response from real articles, citing sources

> **Key feature:** The model is strictly constrained to **only use information from the database** — no hallucination from pre-training knowledge.

---

## 🏗 System Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│    User      │────▶│                  RAG App (:7860)                 │
│  (Browser)   │◀────│                                                  │
└─────────────┘     │  ┌────────────┐  ┌───────────┐  ┌────────────┐  │
                    │  │ transform  │─▶│ retrieve  │─▶│  filter +  │  │
                    │  │  _query()  │  │ _multi()  │  │  top_k()   │  │
                    │  └────────────┘  └─────┬─────┘  └─────┬──────┘  │
                    │                        │              │         │
                    │                        ▼              ▼         │
                    │               ┌──────────────┐  ┌──────────┐   │
                    │               │  ChromaDB    │  │  Ollama   │   │
                    │               │   (:8000)    │  │ (:11434)  │   │
                    │               │  Vector DB   │  │ gemma3:4b │   │
                    │               └──────────────┘  └──────────┘   │
                    └──────────────────────────────────────────────────┘
```

### Services

| Service | Port | Role |
|---------|------|------|
| **rag-app** | `7860` | Gradio Chat UI + RAG Pipeline |
| **ollama** | `11434` | LLM Server (gemma3:4b) |
| **chromadb** | `8001` | ChromaDB Vector Database |

---

## 📁 Project Structure

```
RAG pipeline/
│
├── Docker
│   ├── .env                  # Environment variables
│   ├── .env.example          # Template for new users
│   ├── .dockerignore         # Files excluded from build
│   ├── Dockerfile            # Build image for RAG app
│   ├── docker-compose.yaml   # Orchestrate 3 services
│   └── entrypoint.sh         # Startup script (wait Ollama → pull model → run)
│
├── Python modules
│   ├── config.py             # Configuration, DB connection, model loading
│   ├── query_transform.py    # Transform 1 prompt → 5 diverse queries
│   ├── retriever.py          # Content-based chunk retrieval
│   ├── metadata_filter.py    # Strict category filtering (LLM-based)
│   ├── ranker.py             # Cross-encoder re-ranking (top-K)
│   ├── pipeline.py           # Orchestrate the full RAG flow
│   └── main.py               # 🚀 Entry point — Gradio Chat UI
│
├── Notebook
│   └── main.ipynb            # Notebook (Google Colab) — all-in-one
│
└── requirements.txt          # Python dependencies
```

---

## 💻 System Requirements

### Docker (Recommended)
- **Docker** ≥ 20.10
- **Docker Compose** ≥ 2.0
- **RAM** ≥ 8 GB (gemma3:4b needs ~4 GB RAM)
- **Disk** ≥ 10 GB (model ~3.3 GB + Docker images)

### Local / Colab
- **Python** ≥ 3.10
- **Ollama** installed and running ([ollama.com](https://ollama.com))
- **RAM** ≥ 8 GB

---

## 🚀 Installation Guide

### Option 1: Docker Compose (Recommended)

> The simplest method — just one command.

**Step 1:** Make sure data exists in ChromaDB

```bash
# Check that the chromadb-data volume already exists
docker volume ls | grep chromadb-data
```

> If there's no data yet, run the **Data Pipeline** first to crawl and store articles into ChromaDB.

**Step 2:** Configure (optional)

```bash
# Edit .env if you need to change settings
# Default values work out of the box for Docker
nano .env
```

**Step 3:** Start everything

```bash
# Build and run all services
docker-compose up -d --build
```

**Step 4:** Wait for startup

```bash
# Watch RAG app logs (wait for model pull to finish)
docker logs -f rag-app
```

It's ready when you see:

```
✅ Model gemma3:4b is ready!
🚀 Starting Vietnamese News Assistant...
🚀 Starting on 0.0.0.0:7860 (share=False)
```

**Step 5:** Use it

Open your browser → **http://localhost:7860**

**Stop the system:**

```bash
docker-compose down
```

**Stop and remove everything (including cached models):**

```bash
docker-compose down -v
```

---

### Option 2: Run Locally

**Step 1:** Install Ollama

```bash
# Windows: download from https://ollama.com/download
# Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh
```

**Step 2:** Start Ollama and pull the model

```bash
ollama serve &
ollama pull gemma3:4b
```

**Step 3:** Install Python dependencies

```bash
pip install -r requirements.txt
```

**Step 4:** Configure

```bash
# Set environment variables for local usage
export CHROMA_URL="http://localhost:8001"      # or ngrok URL
export OLLAMA_HOST="http://localhost:11434"
export GRADIO_SHARE="true"                     # Create a public link
```

Or edit the `.env` file directly.

**Step 5:** Run

```bash
python main.py
```

---

### Option 3: Google Colab

Open `main.ipynb` on Google Colab and run the cells in order:

1. **Cell 1.1** — Install Python packages
2. **Cell 1.2** — Install Ollama on Colab
3. **Cell 1.3** — Start Ollama server
4. **Cell 1.4** — Pull gemma3:4b model (~3-5 minutes)
5. **Cells 2-8** — Import, connect DB, define functions
6. **Cell 9** — Test pipeline (debug mode)
7. **Cell 10** — Launch Gradio UI (auto-creates a public link)

> **Note:** On Colab, update `CHROMA_URL` in cell 3 to your ChromaDB ngrok URL.

---

## ⚙ Configuration

All settings are managed through **environment variables** (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_URL` | `http://chromadb:8000` | ChromaDB URL. Docker: `http://chromadb:8000`. Ngrok: `https://xxxx.ngrok-free.app` |
| `CHROMA_AUTH_TOKEN` | `test-token` | ChromaDB authentication token |
| `COLLECTION_NAME` | `news_articles` | Collection name in ChromaDB |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model (SentenceTransformer) |
| `LLM_MODEL` | `gemma3:4b` | LLM model running on Ollama |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio bind address |
| `GRADIO_SERVER_PORT` | `7860` | Gradio port |
| `GRADIO_SHARE` | `false` | Create a public link (Gradio share) |

---

## 🔬 RAG Pipeline Details

```
User: "What's the latest on OpenAI?"
        │
        ▼
┌─ Step 1: transform_query() ─────────────────────────┐
│  Generate 5 diverse queries using LLM:               │
│    [0] What's the latest on OpenAI?                  │
│    [1] OpenAI latest updates news                    │
│    [2] GPT-4 AI recent developments                  │
│    [3] OpenAI vs other AI companies competition      │
│    [4] ChatGPT market and applications               │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 2: retrieve_multi() ──────────────────────────┐
│  Content-based retrieval:                            │
│  • Query ChromaDB with 5 queries → candidate chunks │
│  • Re-score each chunk: cosine_sim(chunk, prompt)   │
│  • Keep chunks with similarity ≥ 0.25               │
│  • Dedup by (link, chunk_id)                         │
│  → 85 chunks → 28 articles (only relevant parts)    │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 3: filter_by_metadata() ──────────────────────┐
│  LLM determines relevant categories:                 │
│    → ["Technology", "Business"]                      │
│  Strict filter: keep only matching categories        │
│    → 28 articles → 8 articles                        │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 4: get_top_k() ──────────────────────────────┐
│  Cross-encoder re-ranking:                           │
│    #1 [3.70] OpenAI and Anthropic CEOs discuss...   │
│    #2 [3.02] Clawdbot creator joins OpenAI          │
│    #3 [0.39] Nvidia invests $30B in OpenAI          │
│    #4 [0.25] OpenAI developing smart speaker        │
│    #5 [-2.07] ...                                   │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌─ Step 5: LLM Generate ─────────────────────────────┐
│  SystemMessage: "You are a Vietnamese news..."      │
│  HumanMessage:                                       │
│    MANDATORY INSTRUCTIONS:                           │
│    1. STRICTLY FORBIDDEN to use external knowledge  │
│    --- START ARTICLES ---                             │
│    [5 full articles with content]                    │
│    --- END ARTICLES ---                              │
│    QUESTION: What's the latest on OpenAI?           │
│    → Answer based ENTIRELY on articles above.       │
└──────────────────────────────────────────────────────┘
        │
        ▼
   🤖 Answer (with article titles + source links cited)
```

### Models Used

| Model | Role | Size |
|-------|------|------|
| `all-MiniLM-L6-v2` | Embedding (vector search) | ~80 MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking (top-K) | ~80 MB |
| `gemma3:4b` | LLM (query transform + category filter + answer) | ~3.3 GB |

---

## 🎮 GPU Support

If you have an **NVIDIA GPU**, uncomment the `deploy` section in `docker-compose.yaml`:

```yaml
ollama:
    image: ollama/ollama:latest
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Then install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Restart services
docker-compose up -d --build
```

> GPU acceleration provides **5-10x faster** LLM inference compared to CPU.

---

## 🔧 Troubleshooting

### 1. `rag-app` can't connect to ChromaDB

```bash
# Check if ChromaDB is running
docker logs chromadb

# Check if the volume has data
docker exec chromadb ls /data
```

**Common causes:**
- Volume `chromadb-data` doesn't exist → Run the Data Pipeline first
- Wrong `CHROMA_URL` → Check the `.env` file

### 2. Ollama fails to pull the model

```bash
# Check logs
docker logs rag-ollama

# Pull manually
docker exec rag-ollama ollama pull gemma3:4b

# Verify model
docker exec rag-ollama ollama list
```

### 3. LLM generates hallucinated/fabricated information

The pipeline is designed to prevent hallucination:
- Context + instructions are placed in `HumanMessage` (not `SystemMessage`)
- Small models (4B) tend to ignore long `SystemMessage` → fixed by putting everything in `HumanMessage`

If it still hallucinates, try:
- Increase `top_k` (more articles for context)
- Lower `content_sim_threshold` (retrieve more chunks)
- Use a larger model (e.g., `gemma3:12b`, `llama3:8b`)

### 4. Can't access Gradio UI

```bash
# Check if the app is running
docker ps | grep rag-app

# Test the port
curl http://localhost:7860
```

### 5. Clean restart

```bash
# Stop all services
docker-compose down

# Rebuild without cache
docker-compose build --no-cache

# Start fresh
docker-compose up -d --build
```

---

## 📝 Notes

- The **Data Pipeline** (separate folder) is responsible for crawling articles and storing them in ChromaDB. The RAG Pipeline only **reads** data from ChromaDB.
- The first run takes **5-10 minutes** to pull the gemma3:4b model (~3.3 GB). Subsequent runs are fast because the model is cached in the Docker volume `ollama-models`.
- The `main.ipynb` file is kept for use on Google Colab. The `.py` files are the modularized version of the notebook.
