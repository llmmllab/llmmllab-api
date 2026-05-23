#!/usr/bin/env bash
#
# test_txt2img.sh — exercise POST /v1/images/generations.
#
# Usage:
#   ./scripts/test_txt2img.sh "a teacup with steam"
#   ./scripts/test_txt2img.sh "a teacup with steam" qwen-image-2512 1024x1024
#
# Env overrides:
#   API_BASE   default http://localhost:8000
#   API_KEY    bearer token; omit for unauth dev endpoints
#   OUT_DIR    where to drop the decoded PNG (default ./out)

set -euo pipefail

PROMPT="${1:-a teacup with steam}"
MODEL="${2:-qwen-image-2512}"
SIZE="${3:-1024x1024}"

API_BASE="${API_BASE:-http://localhost:8000}"
OUT_DIR="${OUT_DIR:-./out}"
mkdir -p "$OUT_DIR"

AUTH_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

TS=$(date +%s)
RESP_FILE="$OUT_DIR/txt2img_${TS}.json"
PNG_FILE="$OUT_DIR/txt2img_${TS}.png"

echo "→ POST $API_BASE/v1/images/generations"
echo "  prompt = $PROMPT"
echo "  model  = $MODEL"
echo "  size   = $SIZE"

curl -sS -X POST "$API_BASE/v1/images/generations" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]}" \
    -d "$(jq -n \
        --arg prompt "$PROMPT" \
        --arg model "$MODEL" \
        --arg size "$SIZE" \
        '{prompt: $prompt, model: $model, size: $size, n: 1}')" \
    -o "$RESP_FILE"

# Pull the first image out and decode it.
B64=$(jq -r '.data[0].b64_json // empty' "$RESP_FILE")
if [[ -z "$B64" ]]; then
    echo "✘ no b64_json in response:" >&2
    cat "$RESP_FILE" >&2
    exit 1
fi

echo "$B64" | base64 -d > "$PNG_FILE"

echo "✓ saved $PNG_FILE ($(wc -c < "$PNG_FILE") bytes)"
echo "  raw response: $RESP_FILE"
