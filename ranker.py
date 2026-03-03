"""
ranker.py — Chọn K bài báo chính xác nhất bằng cross-encoder re-ranking.

Cross-encoder chấm điểm từng cặp (query, full_article_content)
chính xác hơn nhiều so với bi-encoder (vector cosine similarity),
vì nó xem xét ngữ cảnh đầy đủ của cả query lẫn document.
"""

from typing import List, Dict


def get_top_k(
    articles: List[Dict],
    user_prompt: str,
    k: int = 5,
    cross_encoder=None,
) -> List[Dict]:
    """
    Chọn K bài báo chính xác nhất bằng cross-encoder re-ranking.

    Input là pool bài báo ĐÃ được lọc chặt theo category từ filter_by_metadata().

    Args:
        articles: Danh sách bài báo sau filter_by_metadata() — CHỈ bài đúng category
        user_prompt: Câu hỏi gốc của user
        k: Số bài báo muốn giữ lại
        cross_encoder: CrossEncoder model (None → fallback dùng content_similarity)

    Returns:
        Top-K bài báo, mỗi bài có thêm field 'relevance_score'
    """
    if not articles:
        return []

    if len(articles) <= k:
        for a in articles:
            a.setdefault("relevance_score", a.get("content_similarity", 0))
        return articles

    if cross_encoder is not None:
        MAX_CONTENT_LEN = 512
        pairs = [[user_prompt, a["full_content"][:MAX_CONTENT_LEN]] for a in articles]
        scores = cross_encoder.predict(pairs)
        for i, a in enumerate(articles):
            a["relevance_score"] = float(scores[i])
        articles_scored = sorted(
            articles, key=lambda x: x["relevance_score"], reverse=True
        )
    else:
        # Fallback: dùng content_similarity đã tính ở retrieve_multi
        for a in articles:
            a["relevance_score"] = a.get("content_similarity", 0)
        articles_scored = sorted(
            articles, key=lambda x: x["relevance_score"], reverse=True
        )

    return articles_scored[:k]
