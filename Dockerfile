# DNLSSM Research Platform - reproducible container image.
# Pinned Python + pinned dependency versions guarantee the same numerical
# environment across Linux, macOS (via Docker Desktop) and Windows (via WSL2/Docker Desktop).
FROM python:3.11.10-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/dnlssm

# System packages required to build scipy/statsmodels wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt

COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN pip install -e . --no-deps

RUN mkdir -p /opt/dnlssm/experiments_output /opt/dnlssm/data/cache /opt/dnlssm/data/raw

ENTRYPOINT ["python", "-m", "dnlssm.cli"]
CMD ["--help"]
