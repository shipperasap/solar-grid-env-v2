FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "7860"]
