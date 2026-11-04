# MinerU parsing server (mineru-api).
#
# CPU (default): MINERU_EXTRAS=""   MINERU_MODEL_TIER=basic     → ONNX-based, no torch
# GPU (later):   MINERU_EXTRAS=full MINERU_MODEL_TIER=standard  (torch/vLLM wheels bundle CUDA)
#                (or the CPU image + a remote VLM server: set MINERU_VLM_SERVER_URL at runtime)
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG MINERU_VERSION=4.1.7
ARG MINERU_EXTRAS=""
ARG MINERU_MODEL_TIER=basic
ARG MINERU_MODEL_SOURCE=huggingface

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MINERU_TELEMETRY_DISABLED=1 \
    MINERU_HOME=/opt/mineru

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$MINERU_EXTRAS" ]; then pip install "mineru[${MINERU_EXTRAS}]==${MINERU_VERSION}"; \
    else pip install "mineru==${MINERU_VERSION}"; fi

# Bake models into the image to avoid downloads at runtime
RUN mineru-models-download --tier "${MINERU_MODEL_TIER}" --source "${MINERU_MODEL_SOURCE}"

COPY docker/mineru/entrypoint.sh /entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
