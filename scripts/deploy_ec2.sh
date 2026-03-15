#!/usr/bin/env bash

set -Eeuo pipefail

APP_DIR="${APP_DIR:-$(pwd)}"
AWS_REGION="${AWS_REGION:?missing AWS_REGION}"
ECR_REGISTRY="${ECR_REGISTRY:?missing ECR_REGISTRY}"
IMAGE_URI="${IMAGE_URI:?missing IMAGE_URI}"
CONTAINER_NAME="${CONTAINER_NAME:-matchatonic-ai}"
HOST_PORT="${HOST_PORT:-8000}"
APP_PORT="${APP_PORT:-8000}"
HEALTHCHECK_URL="${HEALTHCHECK_URL:-http://127.0.0.1:${HOST_PORT}/}"
BIND_ADDRESS="${BIND_ADDRESS:-0.0.0.0}"
ENV_FILE="${ENV_FILE:-${APP_DIR}/.env}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed"
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws cli is not installed"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "missing env file: $ENV_FILE"
  exit 1
fi

PREVIOUS_IMAGE_ID="$(docker inspect --format '{{.Image}}' "$CONTAINER_NAME" 2>/dev/null || true)"

echo "[deploy] logging in to ECR"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "[deploy] pulling image: $IMAGE_URI"
docker pull "$IMAGE_URI"

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
  "$IMAGE_URI"

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
