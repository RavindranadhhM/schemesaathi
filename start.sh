#!/bin/bash
set -e

# Start Redis in background
redis-server --daemonize yes --loglevel warning
sleep 1
echo "Redis started"

# Start FastAPI
exec uvicorn api.main:app --host 0.0.0.0 --port 7860 --workers 1
