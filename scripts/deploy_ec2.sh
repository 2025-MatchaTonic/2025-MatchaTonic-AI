#!/usr/bin/env bash

set -Eeuo pipefail

APP_DIR="${APP_DIR:-$(pwd)}"
DEPLOY_REF="${DEPLOY_REF:-main}"
IMAGE_NAME="${IMAGE_NAME:-matchatonic-ai}"
CONTAINER_NAME="${CONTAINER_NAME:-matchatonic-ai}"
HOST_PORT="${HOST_PORT:-8000}"
APP_PORT="${APP_PORT:-8000}"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-http://127.0.0.1:${HOST_PORT}/}"
BIND_ADDRESS="${BIND_ADDRESS:-0.0.0.0}"
ENV_FILE="${ENV_FILE:-.env}"

cd "$APP_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "missing env file: $ENV_FILE"
  exit 1
fi

echo "[deploy] using ref: $DEPLOY_REF"
git fetch origin "$DEPLOY_REF"
git checkout "$DEPLOY_REF"
git pull origin "$DEPLOY_REF"

PREVIOUS_IMAGE_ID="$(docker inspect --format '{{.Image}}' "$CONTAINER_NAME" 2>/dev/null || true)"
NEW_TAG="$(date +%Y%m%d%H%M%S)"
NEW_IMAGE="${IMAGE_NAME}:${NEW_TAG}"

echo "[deploy] building image: $NEW_IMAGE"
docker build -t "$NEW_IMAGE" -t "${IMAGE_NAME}:latest" .

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "[deploy] removing existing container"
  docker rm -f "$CONTAINER_NAME"
fi

echo "[deploy] starting new container"
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  -p "${BIND_ADDRESS}:${HOST_PORT}:${APP_PORT}" \
  "$NEW_IMAGE"

echo "[deploy] waiting for health check: $HEALTHCHECK_URL"
for _ in $(seq 1 30); do
  if curl -fsS "$HEALTHCHECK_URL" >/dev/null 2>&1; then
    echo "[deploy] health check passed"
    docker ps --filter "name=${CONTAINER_NAME}"
    exit 0
  fi
  sleep 2
done

echo "[deploy] health check failed"
docker logs "$CONTAINER_NAME" || true
docker rm -f "$CONTAINER_NAME" || true

if [ -n "$PREVIOUS_IMAGE_ID" ]; then
  echo "[deploy] rolling back to previous image"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --env-file "$ENV_FILE" \
    -p "${BIND_ADDRESS}:${HOST_PORT}:${APP_PORT}" \
    "$PREVIOUS_IMAGE_ID"
fi

exit 1
