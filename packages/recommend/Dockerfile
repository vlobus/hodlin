# syntax=docker/dockerfile:1
#
# Image for the recommend domain (T10). Build from the repo root so the whole
# uv workspace is in context:  docker build -f packages/recommend/Dockerfile .
# (docker-compose does this for you).
FROM python:3.14-slim-bookworm

# The uv binary from its official distroless image — no pip bootstrap needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Dependency layer — invalidated only by the lockfile/manifests, not by app
#    source. Copy every workspace manifest so uv can resolve the graph, but not
#    the source yet, so a code change doesn't re-download torch.
COPY pyproject.toml uv.lock README.md ./
COPY packages/contracts/pyproject.toml packages/contracts/
COPY packages/recommend/pyproject.toml packages/recommend/
COPY packages/execute/pyproject.toml packages/execute/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package hodlin-recommend

# 2) App source, then install the workspace packages themselves (fast).
COPY packages/ packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package hodlin-recommend

# alembic.ini resolves its script_location relative to the recommend package,
# and the entrypoint runs migrations from here before starting the app.
WORKDIR /app/packages/recommend

COPY packages/recommend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "hodlin_recommend.main"]
