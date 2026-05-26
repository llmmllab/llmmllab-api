#!/usr/bin/env bash
#
# test_img2img.sh — exercise POST /v1/images/edits (Qwen-Image-Edit-2511).
#
# Usage:
#   ./scripts/test_img2img.sh path/to/photo.png "make it autumn"
#   ./scripts/test_img2img.sh path/to/photo.png "make it autumn" qwen-image-edit-2511 0.75
#
# Env overrides:
#   API_BASE   default http://localhost:8000
#   API_KEY    bearer token; omit for unauth dev endpoints
#   OUT_DIR    where to drop the decoded PNG (default ./out)

set -euo pipefail

INPUT="${1:?path to input image required}"
PROMPT="${2:?edit prompt required}"
MODEL="${3:-qwen-image-edit-2511}"
DENOISE="${4:-0.75}"

API_BASE="${API_BASE:-http://192.168.0.71:9999}"
OUT_DIR="${OUT_DIR:-./out}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
    echo "✘ file not found: $INPUT" >&2
    exit 1
fi

API_KEY=$LLMMLL_AUTH_TOKEN

AUTH_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

TS=$(date +%s)
RESP_FILE="$OUT_DIR/img2img_${TS}.json"
PNG_FILE="$OUT_DIR/img2img_${TS}.png"

# Encode image to base64 directly to a temp file.  Passing the base64
# blob as a ``jq --arg`` argument blows past the OS argv limit at
# ~128 KB; ``--rawfile`` reads the bytes from disk and avoids argv
# pressure entirely.
B64_FILE=$(mktemp -t img2img_b64.XXXXXX)
trap 'rm -f "$B64_FILE"' EXIT
if base64 --help 2>&1 | grep -q -- '-w'; then
    base64 -w0 "$INPUT" > "$B64_FILE"
else
    base64 < "$INPUT" | tr -d '\n' > "$B64_FILE"
fi

# Build the JSON body the same way — but feed the image bytes from
# ``--rawfile`` and stream curl's body from a file too.
BODY_FILE=$(mktemp -t img2img_body.XXXXXX)
trap 'rm -f "$B64_FILE" "$BODY_FILE"' EXIT
jq -n \
    --arg prompt "$PROMPT" \
    --rawfile image "$B64_FILE" \
    --arg model "$MODEL" \
    --argjson denoise "$DENOISE" \
    '{prompt: $prompt, image: $image, model: $model, denoising_strength: $denoise}' \
    > "$BODY_FILE"

echo "→ POST $API_BASE/v1/images/edits"
echo "  input              = $INPUT ($(wc -c < "$INPUT") bytes)"
echo "  prompt             = $PROMPT"
echo "  model              = $MODEL"
echo "  denoising_strength = $DENOISE"

curl -sS -X POST "$API_BASE/v1/images/edits" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
    --max-time 900 \
    --data-binary "@$BODY_FILE" \
    -o "$RESP_FILE"

B64=$(jq -r '.data[0].b64_json // empty' "$RESP_FILE")
if [[ -z "$B64" ]]; then
    echo "✘ no b64_json in response:" >&2
    cat "$RESP_FILE" >&2
    exit 1
fi

echo "$B64" | base64 -d > "$PNG_FILE"

echo "✓ saved $PNG_FILE ($(wc -c < "$PNG_FILE") bytes)"
echo "  raw response: $RESP_FILE"
