#!/usr/bin/env bash
#
# test_img2-3d-parts.sh — exercise POST /v1/images/3d/parts (Hunyuan3D-Part).
#
# Two-step interaction, mirroring test_img2-3d.sh:
#
#   1. POST /v1/images/3d/parts        — submit the input mesh (base64-encoded
#                                        .glb), get back four download URLs
#   2. GET  /v1/images/3d/parts/{file}  — stream each .glb output (decomposed,
#                                        exploded, bbox, gt_bbox) back through
#                                        the api so clients don't need pod
#                                        access
#
# Usage:
#   ./scripts/test_img2-3d-parts.sh path/to/mesh.glb
#   ./scripts/test_img2-3d-parts.sh path/to/mesh.glb 256              # lower-res
#   ./scripts/test_img2-3d-parts.sh path/to/mesh.glb 512 42           # seed
#   ./scripts/test_img2-3d-parts.sh path/to/mesh.glb 512 42 1         # split
#   SPLIT=1 ./scripts/test_img2-3d-parts.sh path/to/mesh.glb          # split via env
#
# Positional args:
#   1. input mesh .glb (required) — typically the output of test_img2-3d.sh
#   2. octree_resolution (default 512; valid 128 or higher)
#   3. seed (optional; empty string skips)
#   4. split (optional; pass 1/true to also export per-part .glb files
#      — each detected part becomes ``<id>_part_NN.glb`` so you can
#      import them as separate objects in Blender / three.js / Unity
#      without splitting the combined decomposed.glb manually)
#
# Env overrides:
#   API_BASE   default http://localhost:8000
#   API_KEY    bearer token; omit for unauth dev endpoints
#   OUT_DIR    where to drop the decoded .glb files + JSON response
#              (default ./out)
#   SPLIT      alternative to the 4th positional arg (any non-empty,
#              non-zero, non-"false" value enables split)

set -euo pipefail

INPUT="${1:?path to input mesh .glb required}"
OCTREE="${2:-512}"
SEED="${3:-}"
SPLIT_RAW="${4:-${SPLIT:-}}"
# ``${var,,}`` (bash 4+ lowercase modifier) breaks on macOS bash 3.2
# with "bad substitution".  Use ``tr`` for portability.
SPLIT_LOWER=$(printf '%s' "$SPLIT_RAW" | tr '[:upper:]' '[:lower:]')
case "$SPLIT_LOWER" in
    1|true|yes|on|y) SPLIT_JSON=true ;;
    *)               SPLIT_JSON=false ;;
esac

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
RESP_FILE="$OUT_DIR/img23d_parts_${TS}.json"

# Base64-encode the input mesh.  Pass via --rawfile to dodge the
# OS argv length cap on multi-MB blobs (same trick as test_img2img.sh).
B64_FILE=$(mktemp -t img23d_parts_b64.XXXXXX)
BODY_FILE=$(mktemp -t img23d_parts_body.XXXXXX)
trap 'rm -f "$B64_FILE" "$BODY_FILE"' EXIT

if base64 --help 2>&1 | grep -q -- '-w'; then
    base64 -w0 "$INPUT" > "$B64_FILE"
else
    base64 < "$INPUT" | tr -d '\n' > "$B64_FILE"
fi

JQ_ARGS=(
    --rawfile mesh "$B64_FILE"
    --argjson octree "$OCTREE"
    --argjson split "$SPLIT_JSON"
)
JQ_EXPR='{mesh_b64: $mesh, octree_resolution: $octree, split: $split}'
if [[ -n "$SEED" ]]; then
    JQ_ARGS+=(--argjson seed "$SEED")
    JQ_EXPR='{mesh_b64: $mesh, octree_resolution: $octree, split: $split, seed: $seed}'
fi

jq -n "${JQ_ARGS[@]}" "$JQ_EXPR" > "$BODY_FILE"

echo "→ POST $API_BASE/v1/images/3d/parts"
echo "  input             = $INPUT ($(wc -c < "$INPUT") bytes)"
echo "  octree_resolution = $OCTREE"
[[ -n "$SEED" ]] && echo "  seed              = $SEED"
echo "  split             = $SPLIT_JSON"
echo "  (XPart can take several minutes; no streaming)"

HTTP_STATUS=$(curl -sS -X POST "$API_BASE/v1/images/3d/parts" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
    --max-time 1800 \
    --data-binary "@$BODY_FILE" \
    -o "$RESP_FILE" \
    -w "%{http_code}")

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "✘ HTTP $HTTP_STATUS from $API_BASE/v1/images/3d/parts" >&2
    echo "  response body:" >&2
    cat "$RESP_FILE" >&2
    echo >&2
    exit 1
fi

if ! jq -e '.id' "$RESP_FILE" >/dev/null 2>&1; then
    echo "✘ server returned 200 but body is not valid JSON or has no id:" >&2
    cat "$RESP_FILE" >&2
    exit 1
fi

ID=$(jq -r '.id' "$RESP_FILE")
ELAPSED=$(jq -r '.elapsed_sec' "$RESP_FILE")
MESH_URL=$(jq -r '.mesh_url // empty' "$RESP_FILE")
EXPLODED_URL=$(jq -r '.exploded_url // empty' "$RESP_FILE")
BBOX_URL=$(jq -r '.bbox_url // empty' "$RESP_FILE")
GT_BBOX_URL=$(jq -r '.gt_bbox_url // empty' "$RESP_FILE")

echo "✓ id           = $ID"
echo "  elapsed_sec  = ${ELAPSED}"

download() {
    local label="$1" url="$2"
    [[ -z "$url" ]] && return
    local out="$OUT_DIR/${ID}_${label}.glb"
    echo "  ↓ GET $API_BASE$url"
    curl -sS "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
        --max-time 120 \
        -o "$out" \
        "$API_BASE$url"
    if [[ -s "$out" ]]; then
        echo "    → $out ( $(wc -c < "$out") bytes)"
    else
        echo "    ✘ empty download: $out" >&2
        return 1
    fi
}

download decomposed "$MESH_URL"
download exploded   "$EXPLODED_URL"
download bbox       "$BBOX_URL"
download gt_bbox    "$GT_BBOX_URL"

# When split=true, also fetch each per-part .glb so the caller has
# them as separate files ready to drop into Blender etc.  Names are
# ``<id>_part_NN.glb`` to match what the runner writes.
if [[ "$SPLIT_JSON" == "true" ]]; then
    N=$(jq -r '.part_urls | length' "$RESP_FILE")
    echo "  parts        = $N"
    for i in $(seq 0 $((N-1))); do
        PART_URL=$(jq -r ".part_urls[$i]" "$RESP_FILE")
        # Mirror the runner's two-digit-pad to match the original filename.
        PART_LABEL=$(printf "part_%02d" "$i")
        download "$PART_LABEL" "$PART_URL"
    done
fi

echo "  raw response: $RESP_FILE"
