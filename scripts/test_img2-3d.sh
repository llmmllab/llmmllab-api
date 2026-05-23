#!/usr/bin/env bash
#
# test_img2-3d.sh — exercise POST /v1/images/3d (TRELLIS pipeline).
#
# Submits a conditioning image, waits for the synchronous TRELLIS run,
# then downloads the resulting .glb (and .ply if requested) via the
# api's /v1/images/3d/{filename} proxy endpoint.
#
# Usage:
#   ./scripts/test_img2-3d.sh path/to/photo.png
#   ./scripts/test_img2-3d.sh path/to/photo.png "mesh,gaussian"
#
# Env overrides:
#   API_BASE   default http://localhost:8000
#   API_KEY    bearer token; omit for unauth dev endpoints
#   OUT_DIR    where to drop the response + downloaded artefacts (default ./out)

set -euo pipefail

INPUT="${1:?path to input image required}"
FORMATS="${2:-mesh}"

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

# base64 -w0 on linux, base64 with no wrapping on mac
if base64 --help 2>&1 | grep -q -- '-w'; then
    B64_IN=$(base64 -w0 "$INPUT")
else
    B64_IN=$(base64 < "$INPUT" | tr -d '\n')
fi

# Convert "mesh,gaussian" → ["mesh","gaussian"]
FORMATS_JSON=$(echo "$FORMATS" | jq -R 'split(",")')

TS=$(date +%s)
RESP_FILE="$OUT_DIR/img23d_${TS}.json"
PREVIEW_FILE="$OUT_DIR/img23d_${TS}_preview.png"

echo "→ POST $API_BASE/v1/images/3d"
echo "  input   = $INPUT ($(wc -c < "$INPUT") bytes)"
echo "  formats = $FORMATS"
echo "  (this can take minutes; TRELLIS doesn't stream)"

curl -sS -X POST "$API_BASE/v1/images/3d" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
    --max-time 1200 \
    -d "$(jq -n \
        --arg img "$B64_IN" \
        --argjson formats "$FORMATS_JSON" \
        '{image_b64: $img, formats: $formats}')" \
    -o "$RESP_FILE"

ID=$(jq -r '.id // empty' "$RESP_FILE")
if [[ -z "$ID" ]]; then
    echo "✘ no id in response:" >&2
    cat "$RESP_FILE" >&2
    exit 1
fi

echo "✓ id           = $ID"
echo "  elapsed_sec  = $(jq -r '.elapsed_sec' "$RESP_FILE")"
echo "  mesh_url     = $(jq -r '.mesh_url // "—"' "$RESP_FILE")"
echo "  gaussian_url = $(jq -r '.gaussian_url // "—"' "$RESP_FILE")"

# Decode preview if returned.
PREVIEW=$(jq -r '.preview_b64 // empty' "$RESP_FILE")
if [[ -n "$PREVIEW" ]]; then
    echo "$PREVIEW" | base64 -d > "$PREVIEW_FILE"
    echo "  preview      = $PREVIEW_FILE"
fi

# Pull each returned artefact through the api's download proxy.  The
# .glb / .ply live on the runner pod's filesystem; the api streams them
# back without needing kubectl access.
download_url() {
    local rel_url="$1"
    [[ -z "$rel_url" || "$rel_url" == "null" ]] && return
    local fname
    fname=$(basename "$rel_url")
    local out_file="$OUT_DIR/$fname"
    echo "  ↓ GET $API_BASE$rel_url"
    curl -sSL "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" "$API_BASE$rel_url" -o "$out_file"
    echo "    → $out_file ($(wc -c < "$out_file") bytes)"
}

download_url "$(jq -r '.mesh_url // empty' "$RESP_FILE")"
download_url "$(jq -r '.gaussian_url // empty' "$RESP_FILE")"

echo "  raw response: $RESP_FILE"
