FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set pip configurations for better reliability
ENV PIP_DEFAULT_TIMEOUT=200
ENV PIP_RETRIES=3
ENV PIP_TIMEOUT=200

# Copy and install requirements with retry mechanism
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    for i in 1 2 3; do \
        pip install --no-cache-dir -r requirements.txt && break || \
        if [ $i -lt 3 ]; then \
            echo "Retry installing requirements: attempt $i" && \
            sleep 5; \
        else \
            exit 1; \
        fi \
    done

# Copy application files
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--browser.serverAddress", "localhost"] 