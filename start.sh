#!/usr/bin/env bash
# =============================================================================
#  start.sh — MenkarAI startup script for Render
# =============================================================================
set -e

# Create runtime directories (ephemeral on free tier — recreated each restart)
mkdir -p downloads vector_db

# Render injects $PORT; fall back to 8501 for local runs
APP_PORT="${PORT:-8501}"

echo "Starting MenkarAI on port $APP_PORT..."

exec streamlit run app.py \
    --server.port="$APP_PORT" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
