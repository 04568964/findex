#build venv from uv.lock
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

#minimal runtime
FROM python:3.13-slim-bookworm

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY deploy/index.bin /data/index.bin

ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH="/data/index.bin" \
    PORT="8000"

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT}"]