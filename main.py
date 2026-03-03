"""
main.py — Gradio Chat UI (Entry Point)

📰 Vietnamese News Assistant
Hỏi đáp về tin tức tiếng Việt bằng RAG pipeline.

Cách chạy:
    python main.py
"""

import traceback
import gradio as gr

from pipeline import rag_pipeline


def chat_fn(message: str, history: list) -> str:
    """Gradio chat callback — dùng RAG pipeline đầy đủ."""
    try:
        chat_history = []
        if history:
            first = history[0]
            if isinstance(first, dict):
                # Gradio mới: list of {"role": ..., "content": ...}
                i = 0
                while i < len(history) - 1:
                    h, a = history[i], history[i + 1]
                    if h.get("role") == "user" and a.get("role") == "assistant":
                        chat_history.append((h["content"], a["content"]))
                        i += 2
                    else:
                        i += 1
            elif isinstance(first, (list, tuple)) and len(first) == 2:
                # Gradio cũ: list of (human, ai) tuples
                chat_history = [(h, a) for h, a in history if h and a]

        return rag_pipeline(
            user_query=message,
            chat_history=chat_history,
            top_k=5,
            n_queries=5,
            distance_threshold=1.5,
            content_sim_threshold=0.25,
            debug=False,
        )
    except Exception as e:
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}"


# ============================================================
# Gradio UI
# ============================================================

demo = gr.ChatInterface(
    fn=chat_fn,
    title="📰 Vietnamese News Assistant",
    description=(
        "Hỏi đáp về tin tức tiếng Việt. "
        "Hệ thống tự động tìm bài báo từ database và trả lời dựa trên nội dung thực tế."
    ),
    examples=[
        "Cho tôi biết tin tức về giá vàng hôm nay",
        "Có tin gì mới về công nghệ AI không?",
        "Tình hình kinh tế Việt Nam hiện tại thế nào?",
        "Tin thể thao mới nhất",
        "Tóm tắt tình hình thế giới",
    ],
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    import os

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"

    print(f"🚀 Starting on {server_name}:{server_port} (share={share})")
    demo.launch(server_name=server_name, server_port=server_port, share=share)
