#!/bin/bash

# Load environment variables
set -a
[ -f .env ] && . .env
set +a

# Default port if not set in environment
PORT="${PORT:-8007}"
HOST="${HOST:-0.0.0.0}"

# Start the application
exec uvicorn app.api.main:app --host "$HOST" --port "$PORT"
