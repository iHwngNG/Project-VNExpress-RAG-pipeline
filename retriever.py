"""
retriever.py — Content-based retrieval từ ChromaDB.

Retrieve chunks theo NỘI DUNG:
1. Multi-query search → thu thập chunks ứng viên
2. Re-score từng chunk bằng cosine similarity với user prompt gốc
3. Chỉ giữ chunks có nội dung thực sự liên quan
4. Dedup theo (link, chunk_id)
5. Group theo bài báo → mỗi bài chỉ chứa các phần liên quan
"""

import numpy as np
from typing import List, Dict


def _cosine_similarity(vec_a, vec_b) -> float:
    """Tính cosine similarity giữa 2 vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve_multi(
    queries: List[str],
    user_prompt: str,
    embedding_model,
    collection,
    max_fetch_per_query: int = 20,
    distance_threshold: float = 1.5,
    content_sim_threshold: float = 0.25,
) -> List[Dict]:
    """
    Retrieve chunks theo NỘI DUNG — content-based retrieval.

    Quy trình:
    1. Dùng multi-query để tìm chunks ứng viên từ ChromaDB
    2. Dedup theo (link, chunk_id) — cùng chunk không lấy 2 lần
    3. RE-SCORE: Tính cosine similarity giữa MỖI chunk và user prompt gốc
       → đảm bảo nội dung chunk thực sự liên quan đến câu hỏi
    4. Chỉ giữ chunks có content_similarity >= ngưỡng
    5. Group theo bài báo → mỗi bài chứa đúng các phần liên quan

    Args:
        queries: Danh sách queries đã transform
        user_prompt: Câu hỏi GỐC của user (dùng để re-score nội dung)
        embedding_model: SentenceTransformer
        collection: ChromaDB collection
        max_fetch_per_query: Số chunks tối đa mỗi query
        distance_threshold: Ngưỡng L2 distance cho multi-query search
        content_sim_threshold: Ngưỡng cosine similarity tối thiểu
            giữa chunk và user_prompt (0.0-1.0). Chunks dưới ngưỡng bị loại.
            Mặc định 0.25 (khá mở, phù hợp cross-lingual).

    Returns:
        List[Dict] các bài báo, mỗi bài chứa CHỈ các chunks
        có nội dung liên quan đến user prompt
    """
    # ── Bước 1: Multi-query search → thu thập chunks ứng viên ──
    chunk_map: Dict[tuple, Dict] = {}

    for query_idx, query in enumerate(queries):
        embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()

        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=max_fetch_per_query,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"⚠️  Query [{query_idx}] error: {e}")
            continue

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # Lọc theo distance threshold, fallback top-3
        relevant = [(i, d) for i, d in enumerate(dists) if d <= distance_threshold]
        if not relevant:
            relevant = [(i, dists[i]) for i in range(min(3, len(dists)))]

        for idx, dist in relevant:
            link = metas[idx].get("link", f"unknown_{query_idx}_{idx}")
            chunk_id = metas[idx].get("chunk_id", idx)
            key = (link, chunk_id)

            if key not in chunk_map:
                chunk_map[key] = {
                    "text": docs[idx],
                    "meta": metas[idx],
                    "best_distance": dist,
                    "matched_queries": [query],
                }
            else:
                chunk_map[key]["best_distance"] = min(
                    chunk_map[key]["best_distance"], dist
                )
                if query not in chunk_map[key]["matched_queries"]:
                    chunk_map[key]["matched_queries"].append(query)

    if not chunk_map:
        return []

    # ── Bước 2: Re-score từng chunk với user prompt gốc ────────
    prompt_emb = embedding_model.encode(user_prompt, convert_to_numpy=True)

    # Embed tất cả chunk texts cùng lúc (batch, nhanh hơn)
    keys_list = list(chunk_map.keys())
    texts_list = [chunk_map[k]["text"] for k in keys_list]
    chunk_embs = embedding_model.encode(texts_list, convert_to_numpy=True)

    # Tính cosine similarity với user prompt cho từng chunk
    scored_chunks = []
    for i, key in enumerate(keys_list):
        sim = _cosine_similarity(prompt_emb, chunk_embs[i])
        chunk_map[key]["content_similarity"] = sim
        scored_chunks.append((key, sim))

    # ── Bước 3: Lọc chunks theo content similarity ─────────────
    filtered_keys = [key for key, sim in scored_chunks if sim >= content_sim_threshold]

    # Fallback: nếu không chunk nào đạt ngưỡng, lấy top-10 cao nhất
    if not filtered_keys:
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        filtered_keys = [key for key, _ in scored_chunks[:10]]

    # ── Bước 4: Group chunks theo bài báo ──────────────────────
    article_chunks: Dict[str, Dict] = {}
    for key in filtered_keys:
        link, chunk_id = key
        chunk_data = chunk_map[key]

        if link not in article_chunks:
            article_chunks[link] = {
                "meta": chunk_data["meta"],
                "best_distance": chunk_data["best_distance"],
                "best_similarity": chunk_data["content_similarity"],
                "matched_queries": list(chunk_data["matched_queries"]),
                "chunks": [],
            }
        else:
            article_chunks[link]["best_distance"] = min(
                article_chunks[link]["best_distance"], chunk_data["best_distance"]
            )
            article_chunks[link]["best_similarity"] = max(
                article_chunks[link]["best_similarity"],
                chunk_data["content_similarity"],
            )
            for q in chunk_data["matched_queries"]:
                if q not in article_chunks[link]["matched_queries"]:
                    article_chunks[link]["matched_queries"].append(q)

        article_chunks[link]["chunks"].append(
            {
                "chunk_id": chunk_id,
                "text": chunk_data["text"],
                "distance": chunk_data["best_distance"],
                "content_similarity": chunk_data["content_similarity"],
            }
        )

    # ── Bước 5: Build articles ─────────────────────────────────
    articles = []
    for link, data in article_chunks.items():
        meta = data["meta"]
        # Sort chunks theo chunk_id để giữ đúng thứ tự bài viết
        sorted_chunks = sorted(data["chunks"], key=lambda c: c["chunk_id"])
        content = "\n\n".join(c["text"] for c in sorted_chunks)

        articles.append(
            {
                "title": meta.get("title", "N/A"),
                "category": meta.get("category", "N/A"),
                "published_date": meta.get("published_date", "N/A"),
                "link": link,
                "full_content": content,
                "distance": data["best_distance"],
                "content_similarity": data["best_similarity"],
                "matched_queries": data["matched_queries"],
                "num_chunks": len(sorted_chunks),
                "chunk_ids": [c["chunk_id"] for c in sorted_chunks],
                "chunk_similarities": [
                    round(c["content_similarity"], 3) for c in sorted_chunks
                ],
            }
        )

    # Sort bài báo theo content similarity cao nhất (giảm dần)
    articles.sort(key=lambda x: x["content_similarity"], reverse=True)
    return articles
