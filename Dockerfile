FROM python:3.10-slim WORKDIR /app COPY 
requirements.txt . RUN pip install --no-cache-dir -r
requirements.txt --extra-index-url 
https://download.pytorch.org/whl/cpu COPY . . ENV
PORT=8080 EXPOSE 8080 CMD exec uvicorn app:app --host
0.0.0.0 --port ${PORT}
