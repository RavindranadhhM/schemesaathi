FROM python:3.11-slim

WORKDIR /app

# System deps including Redis
RUN apt-get update && apt-get install -y \
    build-essential curl redis-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 model during build (CPU mode for HF Spaces)
RUN python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')"

# Copy project files
COPY . .

# Remove local-only files
RUN rm -rf data/raw data/processed venv .env

EXPOSE 7860

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
