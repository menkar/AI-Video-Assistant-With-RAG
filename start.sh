#!/usr/bin/env bash
# =============================================================================
#  start.sh — MenkarAI startup bootstrap script for Render
#
#  This script runs before Streamlit starts to:
#    1. Resolve the correct data directory (persistent disk or local)
#    2. Create required runtime directories
#    3. Export path env vars for the app to consume
#    4. Launch Streamlit on the port Render provides via $PORT
# =============================================================================

set -e   # Exit immediately on any error

echo "============================================================"
echo "  MenkarAI — Starting up..."
echo "============================================================"

# ── Resolve data directories ────────────────────────────────────────────────
# If Render's persistent disk is mounted at /data, use it.
# Otherwise fall back to local directories (dev / free-tier deploys).
if [ -d "/data" ]; then
    export DOWNLOAD_DIR="/data/downloads"
    export VECTOR_DB_DIR="/data/vector_db"
    echo "  Persistent disk detected at /data"
else
    export DOWNLOAD_DIR="downloads"
    export VECTOR_DB_DIR="vector_db"
    echo "  No persistent disk — using local directories"
fi

# ── Create required runtime directories ─────────────────────────────────────
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$VECTOR_DB_DIR"
echo "  Data dirs ready: $DOWNLOAD_DIR | $VECTOR_DB_DIR"

# ── Resolve PORT (Render injects $PORT; default to 8501 for local runs) ─────
APP_PORT="${PORT:-8501}"
echo "  Binding to port: $APP_PORT"

echo "============================================================"
echo "  Launching Streamlit..."
echo "============================================================"

# ── Start Streamlit ──────────────────────────────────────────────────────────
exec streamlit run app.py \
    --server.port="$APP_PORT" \
    --server.address="0.0.0.0" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=500 \
    --browser.gatherUsageStats=false
