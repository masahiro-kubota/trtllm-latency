#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/.uv-cache}"

uv sync --python 3.12
