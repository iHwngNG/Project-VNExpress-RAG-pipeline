"""
query_transform.py — Biến 1 user prompt thành N queries đa dạng.

Kỹ thuật: Multi-Query Retrieval.
Dùng LLM để tạo ra các câu truy vấn tìm kiếm đa dạng,
mỗi query khai thác một góc độ/từ ngữ khác nhau → tăng recall.
"""

import json
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from config import llm


def transform_query(user_prompt: str, n: int = 5) -> List[str]:
    """
    Biến user prompt thành N queries đa dạng bằng LLM.

    Args:
        user_prompt: Câu hỏi gốc từ người dùng
        n: Số queries muốn tạo (mặc định 5)

    Returns:
        List các query strings (bao gồm cả query gốc)
    """
    system_msg = (
        "Bạn là chuyên gia tìm kiếm thông tin tin tức tiếng Việt. "
        "Nhiệm vụ của bạn là tạo ra các câu truy vấn tìm kiếm đa dạng "
        "để tìm bài báo liên quan đến yêu cầu của người dùng.\n\n"
        f"Từ câu hỏi gốc, hãy tạo ra đúng {n} câu truy vấn TÌM KIẾM khác nhau, "
        "mỗi câu khai thác một góc độ hoặc từ khóa khác nhau.\n\n"
        "YÊU CẦU:\n"
        "- Mỗi câu truy vấn ngắn gọn (5-15 từ)\n"
        "- Đa dạng về từ ngữ, không lặp ý\n"
        "- Phù hợp với ngữ cảnh báo chí tiếng Việt\n"
        "- Trả về ĐÚNG định dạng JSON sau, không thêm text khác:\n"
        '{"queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]}'
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Câu hỏi gốc: {user_prompt}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Parse JSON từ response (xử lý cả trường hợp LLM thêm text thừa)
        json_match = re.search(r'\{[^{}]*"queries"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            queries = data.get("queries", [])
        else:
            # Fallback: tách theo dòng nếu không có JSON
            lines = [
                l.strip().lstrip("0123456789.-) ") for l in raw.split("\n") if l.strip()
            ]
            queries = [l for l in lines if len(l) > 5][:n]

    except Exception as e:
        print(f"⚠️  transform_query error: {e}")
        queries = []

    # Luôn bao gồm query gốc và đảm bảo không duplicate
    all_queries = [user_prompt]
    for q in queries:
        q = q.strip()
        if q and q.lower() != user_prompt.lower() and q not in all_queries:
            all_queries.append(q)

    # Nếu LLM không tạo đủ, thêm biến thể đơn giản
    if len(all_queries) < 2:
        all_queries.append(user_prompt + " tin tức")
        all_queries.append(user_prompt + " mới nhất")

    return all_queries[: n + 1]  # Trả về tối đa n+1 (gốc + n)
