# Build context is the project root, e.g.:
#   docker build -f api/Dockerfile -t churn-api .
#   docker run -p 8000:8000 churn-api
FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost / shap wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
