#!/usr/bin/env bash
# Pushes the committed project to a Hugging Face Space (Docker SDK).
#
#   ./deploy/push_to_hf_space.sh <hf-username>/<space-name>
#
# Before the first run: create the Space on huggingface.co (SDK: Docker), and a
# write access token (Settings -> Access Tokens). Git asks for your username and
# the token as the password. Only committed files are sent - .env, data/ and
# models/ never leave this machine.
set -euo pipefail

SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
# a Space needs README.md with its settings at the top; remove the project's
# Readme.md first (macOS file names ignore case, so a copy would land in it)
rm -f "$TMP/Readme.md"
cp "$ROOT/deploy/huggingface/README.md" "$TMP/README.md"

cd "$TMP"
git init -q -b main
git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
git remote add origin "https://huggingface.co/spaces/${SPACE}"
git push --force origin main
echo "Pushed. The Space builds now: https://huggingface.co/spaces/${SPACE}"
