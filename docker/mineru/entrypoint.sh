#!/bin/sh
# Runtime settings (override via compose environment)
#   MINERU_TIER            highest parse tier: flash | basic | standard | advanced | auto  (CPU default: basic)
#   MINERU_API_CONCURRENCY concurrent parse jobs (CPU default: 1)
#   MINERU_VLM_SERVER_URL  remote VLM server (e.g. mineru-openai-server on a GPU node); serves the standard tier remotely
set -e
set -- --host 0.0.0.0 --port 8000 \
  --tier "${MINERU_TIER:-basic}" \
  --concurrency "${MINERU_API_CONCURRENCY:-1}" \
  "$@"
if [ -n "${MINERU_VLM_SERVER_URL:-}" ]; then
  set -- "$@" --vlm-server-url "$MINERU_VLM_SERVER_URL"
fi
exec mineru-api "$@"
