FROM python:3.10-slim

WORKDIR /app

# CPU-only torch wheel keeps the image smaller and installs faster on free tier.
# Using --extra-index-url (not --index-url) so pip can still fall back to normal
# PyPI for everything else, which is more reliable across build environments.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Render (and most platforms) inject a PORT env var and expect the
# container to listen on it.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
