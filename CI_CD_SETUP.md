# CI/CD Setup

## Overview

This repository uses two GitHub Actions workflows:

- `CI`: validates Python syntax, imports the FastAPI app, builds the Docker image, and runs a smoke test
- `CD`: deploys to EC2 only after `CI` succeeds on `main`, or when manually triggered

## Files

- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`
- `scripts/deploy_ec2.sh`

## Required GitHub Secrets

- `EC2_HOST`: public IP or domain of the FastAPI EC2 instance
- `EC2_USERNAME`: SSH user, usually `ubuntu`
- `EC2_SSH_KEY`: private SSH key contents
- `EC2_PORT`: SSH port, usually `22`
- `EC2_APP_DIR`: absolute path to the checked-out repository on EC2

Example:

- `EC2_HOST=43.200.181.53`
- `EC2_USERNAME=ubuntu`
- `EC2_PORT=22`
- `EC2_APP_DIR=/home/ubuntu/venv/2025-MatchaTonic-AI`

## EC2 Prerequisites

- Docker installed and running
- repository cloned at `EC2_APP_DIR`
- `.env` file present in the project root on EC2
- outbound network access available for OpenAI and Pinecone

## Deployment Behavior

The deploy script:

1. fetches and pulls the target ref
2. builds a new Docker image
3. replaces the running container
4. checks `http://127.0.0.1:8000/`
5. rolls back to the previous image if health check fails

## Recommended Branch Policy

- require PR review before merging to `main`
- require `CI` to pass before merge
- keep production deploys limited to `main`
