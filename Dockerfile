FROM python:3.11-slim

# Cài system deps (curl cho healthcheck/entrypoint)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài Python deps trước (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY config.py .
COPY query_transform.py .
COPY retriever.py .
COPY metadata_filter.py .
COPY ranker.py .
COPY pipeline.py .
COPY main.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Gradio mặc định chạy trên port 7860
EXPOSE 7860

# Biến môi trường mặc định
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

ENTRYPOINT ["./entrypoint.sh"]
