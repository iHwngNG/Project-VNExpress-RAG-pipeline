"""
pipeline.py — RAG Pipeline hoàn chỉnh.

Kết hợp toàn bộ:
transform_query → retrieve_multi (content-based) → filter_by_metadata (strict) → get_top_k → LLM

Context + chỉ thị được đưa vào HumanMessage cuối cùng
(tránh attention fade ở model nhỏ như gemma3:4b).
"""

from typing import List, Dict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import embedding_model, collection, llm, cross_encoder
from query_transform import transform_query
from retriever import retrieve_multi
from metadata_filter import filter_by_metadata
from ranker import get_top_k


# ============================================================
# Prompts
# ============================================================

# SystemMessage giữ ngắn gọn — chỉ định vai trò
SYSTEM_PROMPT_SHORT = (
    "Bạn là trợ lý tin tức tiếng Việt. "
    "Bạn chỉ trả lời dựa trên dữ liệu được cung cấp."
)

# Chỉ thị chi tiết — sẽ đưa vào HumanMessage cùng context
RAG_INSTRUCTIONS = (
    "HƯỚNG DẪN BẮT BUỘC:\n"
    "1. NGHIÊM CẤM sử dụng kiến thức bên ngoài. CHỈ dùng thông tin từ các bài báo bên dưới.\n"
    "2. Nếu không có bài báo liên quan, trả lời: 'Không tìm thấy bài báo liên quan trong cơ sở dữ liệu.'\n"
    "3. Luôn trích dẫn tiêu đề bài báo và link nguồn khi trả lời.\n"
    "4. Trả lời bằng tiếng Việt, rõ ràng và súc tích.\n"
    "5. Nếu có nhiều bài báo liên quan, tổng hợp thông tin từ tất cả.\n"
    "6. KHÔNG ĐƯỢC bịa thêm thông tin ngoài những gì có trong các bài báo.\n"
)


# ============================================================
# Helper
# ============================================================


def build_context_text(articles: List[Dict]) -> str:
    """Format danh sách bài báo thành context string cho LLM."""
    if not articles:
        return "[KHÔNG TÌM THẤY BÀI BÁO LIÊN QUAN]"
    parts = []
    for i, a in enumerate(articles):
        n_chunks = a.get("num_chunks", "?")
        parts.append(
            "BÀI BÁO "
            + str(i + 1)
            + " ("
            + str(n_chunks)
            + " phần liên quan)\n"
            + "Tiêu đề: "
            + a["title"]
            + "\n"
            + "Danh mục: "
            + a["category"]
            + "\n"
            + "Ngày đăng: "
            + a["published_date"]
            + "\n"
            + "Link: "
            + a["link"]
            + "\n"
            + "Nội dung:\n"
            + a["full_content"]
            + "\n"
        )
    return "\n".join(parts)


# ============================================================
# Main Pipeline
# ============================================================


def rag_pipeline(
    user_query: str,
    chat_history: list = None,
    top_k: int = 5,
    n_queries: int = 5,
    distance_threshold: float = 1.5,
    content_sim_threshold: float = 0.25,
    debug: bool = False,
) -> str:
    """
    RAG Pipeline đầy đủ:
    1. transform_query    → Tạo N queries từ user prompt
    2. retrieve_multi     → Retrieve chunks theo nội dung (content-based)
    3. filter_by_metadata → Lọc CHẶT theo category
    4. get_top_k          → Chọn K bài chính xác nhất
    5. LLM Generate       → Trả lời dựa trên bài báo
    """
    if debug:
        print(f"\n{'='*60}")
        print(f"📝 User query: {user_query}")

    # ── STEP 1: Transform query ──────────────────────────────
    queries = transform_query(user_query, n=n_queries)
    if debug:
        print(f"\n🔄 Step 1 — Queries ({len(queries)}):")
        for i, q in enumerate(queries):
            print(f"  [{i}] {q}")

    # ── STEP 2: Retrieve (content-based, chunk-level) ────────
    articles = retrieve_multi(
        queries,
        user_prompt=user_query,
        embedding_model=embedding_model,
        collection=collection,
        max_fetch_per_query=20,
        distance_threshold=distance_threshold,
        content_sim_threshold=content_sim_threshold,
    )
    if debug:
        print(f"\n🗃️  Step 2 — Retrieved: {len(articles)} bài báo (content-based)")
        total_chunks = sum(a.get("num_chunks", 0) for a in articles)
        print(f"            Tổng chunks liên quan: {total_chunks}")
        for a in articles[:5]:
            print(
                f"              [sim={a['content_similarity']:.3f}] "
                f"{a['title']} ({a['num_chunks']} chunks)"
            )

    # ── STEP 3: Filter by metadata (luôn strict) ─────────────
    filtered, relevant_cats = filter_by_metadata(articles, user_query)
    if debug:
        print(f"\n🏷️  Step 3 — Relevant categories: {relevant_cats}")
        print(f"            After filter: {len(filtered)} bài (từ {len(articles)} bài)")

    # ── STEP 4: Top-K từ pool đã lọc category ────────────────
    top_articles = get_top_k(filtered, user_query, k=top_k, cross_encoder=cross_encoder)
    if debug:
        print(f"\n🏆 Step 4 — Top-{top_k}:")
        for a in top_articles:
            print(
                f"  [score={a.get('relevance_score', 0):.4f}] "
                f"[{a['category']}] {a['title']} ({a['num_chunks']} chunks)"
            )

    # ── STEP 5: Generate ─────────────────────────────────────
    context_text = build_context_text(top_articles)

    messages = [SystemMessage(content=SYSTEM_PROMPT_SHORT)]

    if chat_history:
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=str(human_msg)))
            messages.append(AIMessage(content=str(ai_msg)))

    final_prompt = (
        RAG_INSTRUCTIONS
        + "\n--- BẮT ĐẦU CÁC BÀI BÁO ---\n"
        + context_text
        + "\n--- KẾT THÚC CÁC BÀI BÁO ---\n\n"
        + "CÂU HỎI CỦA TÔI: "
        + user_query
        + "\n\n"
        + "Hãy trả lời DỰA HOÀN TOÀN vào các bài báo trên. "
        + "KHÔNG được dùng kiến thức bên ngoài."
    )
    messages.append(HumanMessage(content=final_prompt))

    if debug:
        print(f"\n📤 Final HumanMessage ({len(final_prompt)} chars)")
        print(f"   400 ký tự đầu: {final_prompt[:400]}")
        print(f"   ...")
        print(f"   200 ký tự cuối: {final_prompt[-200:]}")
        print(f"{'='*60}\n")

    response = llm.invoke(messages)
    return response.content
