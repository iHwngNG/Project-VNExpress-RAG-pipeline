#!/bin/bash
set -e

echo "⏳ Đợi Ollama server khởi động..."
until curl -s "${OLLAMA_HOST:-http://ollama:11434}" > /dev/null 2>&1; do
    echo "   ... đang đợi Ollama tại ${OLLAMA_HOST:-http://ollama:11434}"
    sleep 2
done
echo "✅ Ollama server đang chạy!"

# Pull model nếu chưa có
MODEL="${LLM_MODEL:-gemma3:4b}"
echo "⏳ Kiểm tra model ${MODEL}..."

# Gọi API pull — nếu model đã có sẵn sẽ rất nhanh
curl -s "${OLLAMA_HOST:-http://ollama:11434}/api/pull" \
    -d "{\"name\": \"${MODEL}\"}" \
    --max-time 600 \
    > /dev/null 2>&1 || true

echo "✅ Model ${MODEL} đã sẵn sàng!"

# Chạy ứng dụng
echo "🚀 Khởi động Vietnamese News Assistant..."
exec python main.py
