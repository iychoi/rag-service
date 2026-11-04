#!/bin/sh
set -e
alembic upgrade head
exec uvicorn --factory rag_service.main:create_app --host 0.0.0.0 --port 8000 "$@"
