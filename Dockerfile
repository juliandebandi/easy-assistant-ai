# syntax=docker/dockerfile:1

# ---- builder ---------------------------------------------------------------
# The astral-sh/uv image ships the `uv` binary on top of a normal Python base image, so
# dependency resolution/install uses the exact same tool and lockfile as local dev — no
# separate "how does CI/prod install deps" logic to keep in sync with pyproject.toml.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# COMPILE_BYTECODE: pay the .pyc compilation cost once at build time instead of on every
# container's first cold start. LINK_MODE=copy: uv defaults to hardlinking from its cache for
# speed, which assumes cache and target share a filesystem — not a safe assumption across a
# Docker layer/mount boundary, so this forces plain copies instead of risking a broken link.

WORKDIR /app

# Dependencies are installed in their own layer, from only the two files that actually
# determine them (pyproject.toml, uv.lock), *before* the application source is copied in.
# Docker caches layers by content hash: touching a .py file won't invalidate this layer, so a
# code-only change rebuilds in seconds instead of re-resolving/reinstalling ~90 packages.
# --no-install-project: install dependencies only, not the app itself — that happens in the
# second `uv sync` below, once the source is actually present, so the editable install points
# at real files instead of an empty tree.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime ----------------------------------------------------------------
# A plain python-slim image for the final stage: uv itself (and its cache mount, and pip) are
# build-time-only tools with no reason to exist in the image that actually gets deployed and
# scanned for vulnerabilities.
FROM python:3.14-slim-bookworm AS runtime

# Running as a non-root user is the one Docker hardening step that costs nothing: if a
# dependency vulnerability ever allows code execution inside this container, it executes as
# an unprivileged user rather than root.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

# The builder's venv is copied wholesale rather than re-running `uv sync` in this stage —
# prepending it to PATH is enough to make `python`/`uvicorn`/`alembic` resolve to the venv's
# copies without an explicit `uv run` wrapper at container runtime.
ENV PATH="/app/.venv/bin:$PATH"

USER app
EXPOSE 8000

CMD ["uvicorn", "backend.interfaces.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
