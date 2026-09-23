#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f .env || { printf 'Run python3 scripts/setup_env.py first.\n' >&2; exit 1; }
if ! source_status="$(git status --porcelain --untracked-files=all)"; then
    printf 'Cannot verify Git source status.\n' >&2
    exit 1
fi
test -z "$source_status" || { printf 'Deploy only a clean committed source tree.\n' >&2; exit 1; }
git merge-base --is-ancestor HEAD origin/main || { printf 'Release must be merged into origin/main.\n' >&2; exit 1; }
if ! RELEASE_COMMIT="$(git rev-parse --verify HEAD)"; then
    printf 'Cannot resolve release commit.\n' >&2
    exit 1
fi
[[ "$RELEASE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf 'Invalid release commit.\n' >&2; exit 1; }
export RELEASE_COMMIT
export RELEASE_TAG="${RELEASE_COMMIT:0:12}"
export RELEASE_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker compose build
IMAGE_ID="$(docker image inspect "ine-ocr:${RELEASE_TAG}" --format '{{.Id}}')"
export RELEASE_ARTIFACT_SHA256="${IMAGE_ID#sha256:}"
docker compose up -d --no-build --wait --wait-timeout 180
python3 scripts/smoke_test.py
python3 scripts/deployment_receipt.py
