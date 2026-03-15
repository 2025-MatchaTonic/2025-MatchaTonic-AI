# CI/CD Setup

## Overview

This repository uses two GitHub Actions workflows:

- `CI`: validates Python syntax, imports the FastAPI app, builds the Docker image, and runs a smoke test
- `CD`: after `CI` succeeds on `main`, it builds a production image, pushes it to Amazon ECR, then updates the EC2 container by pulling that exact image

## Files

- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`
- `scripts/deploy_ec2.sh`

## Required GitHub Secrets

### AWS

- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `ECR_REPOSITORY`

`ECR_REPOSITORY` must be the full repository URI.

Example:

- `AWS_REGION=ap-northeast-2`
- `ECR_REPOSITORY=123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/matchatonic-ai`

### EC2 SSH

- `EC2_HOST`
- `EC2_USERNAME`
- `EC2_SSH_KEY`
- `EC2_PORT`
- `EC2_APP_DIR`

Example:

- `EC2_HOST=43.200.181.53`
- `EC2_USERNAME=ubuntu`
- `EC2_PORT=22`
- `EC2_APP_DIR=/home/ubuntu/venv/2025-MatchaTonic-AI`

## EC2 Prerequisites

- Docker installed and running
- AWS CLI installed
- the EC2 instance can authenticate to ECR
- `.env` file present at `${EC2_APP_DIR}/.env`
- outbound network access available for OpenAI, Pinecone, and ECR

## Recommended EC2 Authentication to ECR

Use an EC2 IAM role with ECR pull permissions instead of long-lived AWS keys on the server.

Minimum permissions:

- `ecr:GetAuthorizationToken`
- `ecr:BatchGetImage`
- `ecr:GetDownloadUrlForLayer`
- `ecr:BatchCheckLayerAvailability`

## Deployment Behavior

The deploy workflow:

1. checks out the repository
2. authenticates to AWS
3. builds a Docker image
4. pushes both `${GITHUB_SHA}` and `latest` tags to ECR
5. SSHes into EC2
6. logs in to ECR from EC2
7. pulls the exact `${GITHUB_SHA}` image
8. replaces the running container
9. checks `http://127.0.0.1:8000/`
10. rolls back to the previous image if health check fails

## Why This Is Better Than EC2 Build Deploys

- EC2 no longer needs the latest source code to deploy
- the exact image tested in CI is the one deployed to production
- bootstrap failures caused by missing server-side scripts are removed
- deploys become faster and more reproducible

## Recommended Branch Policy

- require PR review before merging to `main`
- require `CI` to pass before merge
- keep production deploys limited to `main`
