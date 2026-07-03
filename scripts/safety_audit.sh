#!/usr/bin/env bash
set -euo pipefail

printf "\n=== SHEEDIVA SAFE REPO AUDIT ===\n"
printf "This script only reads files and prints findings. It does not change anything.\n\n"

printf "[1/7] Git remote:\n"
git remote -v || true

printf "\n[2/7] Git status:\n"
git status --short || true

printf "\n[3/7] Hidden files at repo root:\n"
find . -maxdepth 1 -name ".*" -print | sort || true

printf "\n[4/7] Possible secret/key files:\n"
find . \
  -name ".env" -o \
  -name ".env.*" -o \
  -name "*.pem" -o \
  -name "*.key" -o \
  -name "id_rsa" -o \
  -name "id_ed25519" \
  | sort || true

printf "\n[5/7] GitHub workflows:\n"
find .github/workflows -type f -maxdepth 2 2>/dev/null | sort || true

printf "\n[6/7] Suspicious network/token patterns, review manually:\n"
grep -RInE "(webhook|telegram|bot_token|api_key|secret|password|ngrok|localtunnel|requests\.post|fetch\(|curl .*bash|base64 -d|eval\(|exec\()" . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=node_modules \
  --exclude-dir=dist \
  --exclude-dir=build \
  2>/dev/null | head -200 || true

printf "\n[7/7] Python dependency files:\n"
find . -maxdepth 3 \( -name "requirements*.txt" -o -name "pyproject.toml" -o -name "uv.lock" -o -name "poetry.lock" \) -print | sort || true

printf "\n=== DONE ===\n"
printf "Review any findings before running unknown scripts, workflows, or deployments.\n"
