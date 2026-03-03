"""
config.py — Cấu hình, kết nối ChromaDB, load models.

Tất cả constants và shared objects (collection, embedding_model, llm, cross_encoder)
được khởi tạo ở đây và import bởi các module khác.

Cấu hình đọc từ biến môi trường (environment variables),
phù hợp cho cả local lẫn Docker.
"""

import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama

# ============================================================
# CẤU HÌNH — đọc từ ENV, có giá trị mặc định
# ============================================================
CHROMA_URL = os.environ.get("CHROMA_URL", "http://chromadb:8000")
CHROMA_AUTH_TOKEN = os.environ.get("CHROMA_AUTH_TOKEN", "test-token")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "news_articles")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")

# ============================================================
# Kết nối ChromaDB
# ============================================================
# Phân tích URL để xác định ssl và port
_use_ssl = CHROMA_URL.startswith("https://")
_port = 443 if _use_ssl else 8000

# Nếu URL có port tường minh, dùng port đó
import re as _re

_port_match = _re.search(r":(\d+)$", CHROMA_URL.rstrip("/").split("://")[-1])
if _port_match:
    _port = int(_port_match.group(1))

# Lấy host (bỏ scheme và port)
_host = CHROMA_URL.rstrip("/")

settings = Settings(
    anonymized_telemetry=False,
    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
    chroma_client_auth_credentials=CHROMA_AUTH_TOKEN,
)

client = chromadb.HttpClient(host=_host, port=_port, ssl=_use_ssl, settings=settings)
heartbeat = client.heartbeat()
print(f"✅ ChromaDB connected | Heartbeat: {heartbeat}")

collection = client.get_collection(name=COLLECTION_NAME)
total_docs = collection.count()
print(f"📊 Collection '{COLLECTION_NAME}': {total_docs} chunks")

# ============================================================
# Load Models
# ============================================================
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
print(
    f"✅ Embedding: {EMBEDDING_MODEL} | dim={embedding_model.get_sentence_embedding_dimension()}"
)

llm = ChatOllama(model=LLM_MODEL, temperature=0.1, base_url=OLLAMA_HOST)
print(f"✅ LLM: {LLM_MODEL} (via {OLLAMA_HOST})")

# Cross-encoder để re-rank (Top-K)
cross_encoder = None
try:
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("✅ Cross-encoder: ms-marco-MiniLM-L-6-v2")
except Exception as e:
    print(f"⚠️  Cross-encoder không khả dụng: {e}")
