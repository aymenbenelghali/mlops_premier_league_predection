#!/usr/bin/env bash
# Build the Docker image (Linux / macOS / WSL)
set -euo pipefail
TAG=${1:-premier_league_prediction:latest}
# Use buildkit if available for better caching
DOCKER_BUILDKIT=1 docker build --progress=plain -t "$TAG" .

echo "Built image: $TAG"
