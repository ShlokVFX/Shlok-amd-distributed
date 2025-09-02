#!/usr/bin/env bash
# Helper to publish this local clone to your own GitHub repository.
# Usage:
# 1) Create a new empty repo on GitHub (via UI or `gh repo create <name> --private --confirm`).
# 2) Run this script with the new remote URL or use GH CLI.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <git-remote-url> [branch]"
  echo "Example: $0 git@github.com:youruser/reference-kernels.git main"
  exit 1
fi

REMOTE_URL=$1
BRANCH=${2:-main}

echo "Adding remote 'origin' -> $REMOTE_URL"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo "Pushing branch $BRANCH to origin (will create remote branch)"
git push -u origin "$BRANCH"

echo "Done. Your local repository is now connected to $REMOTE_URL"
