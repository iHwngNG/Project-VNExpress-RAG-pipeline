"""
metadata_filter.py — Lọc CHẶT bài báo theo category phù hợp.

Dùng LLM để xác định categories nào phù hợp với prompt,
sau đó chỉ giữ lại bài thuộc đúng categories đó.
"""

import json
import re
from typing import List, Dict, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config import llm


def determine_relevant_categories(
    user_prompt: str, available_categories: List[str]
) -> List[str]:
    """
    Dùng LLM để xác định categories nào trong DB phù hợp với prompt của user.

    Args:
        user_prompt: Câu hỏi gốc từ người dùng
        available_categories: Danh sách categories thực tế trong DB

    Returns:
        List categories phù hợp (subset của available_categories)
    """
    if not available_categories:
        return []

    cats_str = ", ".join(f'"{c}"' for c in available_categories)

    system_msg = (
        "Bạn là hệ thống phân loại tin tức. "
        "Nhiệm vụ là chọn các danh mục (category) tin tức PHÙ HỢP NHẤT với câu hỏi của người dùng.\n\n"
        f"Các danh mục có sẵn trong database: {cats_str}\n\n"
        "Hãy chọn MỘT HOẶC NHIỀU danh mục phù hợp nhất. "
        "Nếu câu hỏi rất chung chung, có thể trả về tất cả.\n"
        "Trả về ĐÚNG định dạng JSON sau, không thêm text khác:\n"
        '{"relevant_categories": ["category1", "category2"]}'
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Câu hỏi người dùng: {user_prompt}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        json_match = re.search(r'\{[^{}]*"relevant_categories"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            selected = data.get("relevant_categories", [])
            valid = [c for c in selected if c in available_categories]
            return valid if valid else available_categories
    except Exception as e:
        print(f"⚠️  determine_relevant_categories error: {e}")

    return available_categories  # Fallback


def filter_by_metadata(
    articles: List[Dict],
    user_prompt: str,
) -> Tuple[List[Dict], List[str]]:
    """
    Lọc CHẶT bài báo — chỉ giữ bài thuộc category phù hợp với prompt.

    Quy trình:
    1. Lấy danh sách categories từ danh sách bài báo hiện có
    2. Dùng LLM xác định categories nào phù hợp với prompt
    3. Chỉ giữ lại bài thuộc đúng category đó (strict)
    4. Fallback: giữ tất cả nếu không có bài nào match (edge case)

    Returns:
        (filtered_articles, relevant_categories)
        filtered_articles: CHỈ các bài đúng category — đưa thẳng vào get_top_k()
    """
    if not articles:
        return [], []

    available_categories = list(
        set(
            a["category"].strip()
            for a in articles
            if a.get("category") and a["category"].strip() not in ("", "N/A")
        )
    )

    if not available_categories:
        return articles, []

    relevant_cats = determine_relevant_categories(user_prompt, available_categories)

    # Lọc CHẶT: chỉ giữ bài thuộc category phù hợp
    matched = [a for a in articles if a.get("category", "").strip() in relevant_cats]

    # Fallback: nếu không có bài nào match → dùng tất cả để tránh trả về rỗng
    if not matched:
        print(
            f"⚠️  Không có bài nào thuộc categories {relevant_cats}. Fallback: dùng tất cả."
        )
        return articles, relevant_cats

    return matched, relevant_cats
