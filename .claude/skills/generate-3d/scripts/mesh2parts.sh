#!/usr/bin/env bash
#
# mesh2parts.sh — exercise POST /v1/images/3d/parts (Hunyuan3D-Part).
#
# Two-step interaction, mirroring img2-3d.sh:
#
#   1. POST /v1/images/3d/parts        — submit the input mesh (base64-encoded
#                                        .glb), get back per-part download URLs
#   2. GET  /v1/images/3d/parts/{file}  — stream each .glb output (per-part files
#                                        plus exploded / bbox / gt_bbox views)
#                                        back through the api so clients don't
#                                        need pod access
#
# Each detected part is exported as its own ``<id>_part_NN.glb`` so you can
# drop them straight into Blender / three.js / Unity as separate objects
# without post-processing the combined decomposed.glb.  This script no
# longer takes a ``split`` toggle — split mode is always on; the assembled
# ``<id>_decomposed.glb`` is still produced alongside the per-part files
# so callers who want the joined mesh still have it.
#
# Usage:
#   ./mesh2parts.sh path/to/mesh.glb
#   ./mesh2parts.sh path/to/mesh.glb 256              # lower-res
#   ./mesh2parts.sh path/to/mesh.glb 512 42           # seed
#
# Positional args:
#   1. input mesh .glb (required) — typically the output of img2-3d.sh
#   2. octree_resolution (default 512; valid 128 or higher)
#   3. seed (optional; empty string skips)
#
# Env overrides:
#   API_BASE   default http://localhost:8000
#   API_KEY    bearer token; omit for unauth dev endpoints
#   OUT_DIR    where to drop the decoded .glb files + JSON response
#              (default ./out)

set -euo pipefail

INPUT="${1:?path to input mesh .glb required}"
OCTREE="${2:-512}"
SEED="${3:-}"

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
RESP_FILE="$OUT_DIR/mesh2parts_${TS}.json"

# Base64-encode the input mesh.  Pass via --rawfile to dodge the
# OS argv length cap on multi-MB blobs (same trick as img2img.sh).
B64_FILE=$(mktemp -t mesh2parts_b64.XXXXXX)
BODY_FILE=$(mktemp -t mesh2parts_body.XXXXXX)
trap 'rm -f "$B64_FILE" "$BODY_FILE"' EXIT

if base64 --help 2>&1 | grep -q -- '-w'; then
    base64 -w0 "$INPUT" > "$B64_FILE"
else
    base64 < "$INPUT" | tr -d '\n' > "$B64_FILE"
fi

# ``split: true`` always — callers want per-part files.  The runner
# also returns the assembled decomposed/exploded/bbox views regardless.
JQ_ARGS=(
    --rawfile mesh "$B64_FILE"
    --argjson octree "$OCTREE"
)
JQ_EXPR='{mesh_b64: $mesh, octree_resolution: $octree, split: true}'
if [[ -n "$SEED" ]]; then
    JQ_ARGS+=(--argjson seed "$SEED")
    JQ_EXPR='{mesh_b64: $mesh, octree_resolution: $octree, split: true, seed: $seed}'
fi

jq -n "${JQ_ARGS[@]}" "$JQ_EXPR" > "$BODY_FILE"

echo "→ POST $API_BASE/v1/images/3d/parts"
echo "  input             = $INPUT ($(wc -c < "$INPUT") bytes)"
echo "  octree_resolution = $OCTREE"
[[ -n "$SEED" ]] && echo "  seed              = $SEED"
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

# Per-part .glb files.  Names are ``<id>_part_NN.glb`` to match what
# the runner writes, so re-running the script idempotently overwrites
# in place.
N=$(jq -r '.part_urls | length' "$RESP_FILE")
echo "  parts        = $N"
for i in $(seq 0 $((N-1))); do
    PART_URL=$(jq -r ".part_urls[$i]" "$RESP_FILE")
    # Mirror the runner's two-digit-pad to match the original filename.
    PART_LABEL=$(printf "part_%02d" "$i")
    download "$PART_LABEL" "$PART_URL"
done

echo "  raw response: $RESP_FILE"
