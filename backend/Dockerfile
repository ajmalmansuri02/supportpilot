# Build from the repo root so the help-centre docs are included:
#   docker build -f backend/Dockerfile .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --no-dev --no-install-project --extra vertex
COPY backend/app ./app
RUN uv sync --no-dev --extra vertex
COPY data/docs /app/data/docs
ENV PATH="/app/backend/.venv/bin:$PATH"
EXPOSE 8000
# Cloud Run tells the container which port to use through $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
