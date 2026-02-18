#!/bin/bash
echo "========================================"
echo "  PDF AI Extractor - Docker Start"
echo "========================================"
echo ""
echo "Starting with Docker Compose..."
echo "(First run will download models — may take 10-15 minutes)"
echo ""
docker compose up --build
