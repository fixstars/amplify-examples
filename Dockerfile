FROM ghcr.io/astral-sh/uv:debian-slim

LABEL maintainer="Yoshiki Matsuda <y_matsuda@fixstars.com>"

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglib2.0-0 \
    git \
    jq \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.10 3.11 3.12 3.13

ENV UV_PROJECT_ENVIRONMENT=.venv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN git config --global --add safe.directory '*'
