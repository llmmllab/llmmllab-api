#!/usr/bin/env bash
#
# test_rembg.sh — exercise POST /v1/images/remove-bg (briaai/RMBG-2.0).
#
# Sends a base64-encoded image to the api, receives the alpha mask and
# (by default) an alpha-composited transparent PNG, decodes both to
# disk, and prints the relative download URL the server reports for
# the cached cutout.
#
# Usage:
#   ./scripts/test_rembg.sh path/to/photo.png
#   ./scripts/test_rembg.sh path/to/photo.png 1                 # mask-only
#   MASK_ONLY=1 ./scripts/test_rembg.sh path/to/photo.png
#
# Env overrides:
#   API_BASE   default http://localhost:8000
#   API_KEY    bearer token; omit for unauth dev endpoints
#   OUT_DIR    where to drop the decoded PNGs (default ./out)
#   SIZE       optional integer — square edge for the model resize
#              (default 1024).  Mask is upsampled back to source res
#              regardless.

set -euo pipefail

INPUT="${1:?path to input image required}"
MASK_ONLY="${2:-${MASK_ONLY:-0}}"

API_BASE="${API_BASE:-http://localhost:8000}"
OUT_DIR="${OUT_DIR:-./out}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
    echo "✘ file not found: $INPUT" >&2
    exit 1
fi

AUTH_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

TS=$(date +%s)
RESP_FILE="$OUT_DIR/rembg_${TS}.json"
MASK_FILE="$OUT_DIR/rembg_${TS}_mask.png"
CUTOUT_FILE="$OUT_DIR/rembg_${TS}_cutout.png"

# Stream the base64 blob and the JSON body via temp files to avoid
# the argv length limit (see test_img2img.sh for the same pattern).
B64_FILE=$(mktemp -t rembg_b64.XXXXXX)
BODY_FILE=$(mktemp -t rembg_body.XXXXXX)
trap 'rm -f "$B64_FILE" "$BODY_FILE"' EXIT

if base64 --help 2>&1 | grep -q -- '-w'; then
    base64 -w0 "$INPUT" > "$B64_FILE"
else
    base64 < "$INPUT" | tr -d '\n' > "$B64_FILE"
fi

if [[ "$MASK_ONLY" == "1" || "$MASK_ONLY" == "true" ]]; then
    MASK_ONLY_JSON=true
else
    MASK_ONLY_JSON=false
fi

JQ_ARGS=(
    --rawfile image "$B64_FILE"
    --argjson mask_only "$MASK_ONLY_JSON"
)
JQ_EXPR='{image: $image, mask_only: $mask_only}'
if [[ -n "${SIZE:-}" ]]; then
    JQ_ARGS+=(--argjson size "$SIZE")
    JQ_EXPR='{image: $image, mask_only: $mask_only, size: $size}'
fi

jq -n "${JQ_ARGS[@]}" "$JQ_EXPR" > "$BODY_FILE"

echo "→ POST $API_BASE/v1/images/remove-bg"
echo "  input      = $INPUT ($(wc -c < "$INPUT") bytes)"
echo "  mask_only  = $MASK_ONLY_JSON"
[[ -n "${SIZE:-}" ]] && echo "  size       = $SIZE"

curl -sS -X POST "$API_BASE/v1/images/remove-bg" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
    --max-time 300 \
    --data-binary "@$BODY_FILE" \
    -o "$RESP_FILE"

# Surface server errors before trying to decode the payload.
if ! jq -e '.id' "$RESP_FILE" >/dev/null 2>&1; then
    echo "✘ server returned an error:" >&2
    cat "$RESP_FILE" >&2
    exit 1
fi

MASK_B64=$(jq -r '.mask_b64' "$RESP_FILE")
TRANSPARENT_B64=$(jq -r '.transparent_b64 // empty' "$RESP_FILE")
CUTOUT_URL=$(jq -r '.cutout_url // empty' "$RESP_FILE")
ELAPSED=$(jq -r '.elapsed_sec' "$RESP_FILE")
WIDTH=$(jq -r '.width' "$RESP_FILE")
HEIGHT=$(jq -r '.height' "$RESP_FILE")
ID=$(jq -r '.id' "$RESP_FILE")

echo "$MASK_B64" | base64 -d > "$MASK_FILE"
echo "✓ saved $MASK_FILE ($(wc -c < "$MASK_FILE") bytes)"

if [[ -n "$TRANSPARENT_B64" ]]; then
    echo "$TRANSPARENT_B64" | base64 -d > "$CUTOUT_FILE"
    echo "✓ saved $CUTOUT_FILE ($(wc -c < "$CUTOUT_FILE") bytes)"
fi

echo
echo "  id          = $ID"
echo "  size        = ${WIDTH}x${HEIGHT}"
echo "  elapsed_sec = ${ELAPSED}"
if [[ -n "$CUTOUT_URL" ]]; then
    echo "  download    = $API_BASE$CUTOUT_URL"
fi
echo "  raw response: $RESP_FILE"
