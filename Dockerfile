FROM python:3.10-slim

WORKDIR /app

# CPU-only torch wheel keeps the image smaller and installs faster on free tier
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects the PORT env var (usually 8080) and expects the
# container to listen on it.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
