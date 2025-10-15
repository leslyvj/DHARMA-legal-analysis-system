#!/bin/bash
set -e

echo "Creating DHARMA project directories..."

mkdir -p dharma_project/backend
mkdir -p dharma_project/frontend
mkdir -p dharma_project/kb_raw
mkdir -p dharma_project/kb_processed
mkdir -p dharma_project/logs
mkdir -p dharma_project/config
mkdir -p dharma_project/tests

# placeholder files
touch dharma_project/backend/app.py
touch dharma_project/frontend/streamlit_app.py
touch dharma_project/config/.env.example
touch dharma_project/kb_raw/README.md
touch dharma_project/kb_processed/README.md

echo "Directory structure created under ./dharma_project"
