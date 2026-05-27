#!/usr/bin/env bash
#
# img2img.sh — exercise POST /v1/images/edits (Qwen-Image-Edit-2511).
#
# Usage:
#   ./scripts/img2img.sh path/to/photo.png "make it autumn"
#   ./scripts/img2img.sh path/to/photo.png "make it autumn" qwen-image-edit-2511 0.75
#
# Env overrides:
#   API_BASE         default http://localhost:8000
#   API_KEY          bearer token; omit for unauth dev endpoints
#   OUT_DIR          where to drop the decoded PNG (default ./out)
#   NEGATIVE_PROMPT  things to exclude from the edit.  Same idea as
#                    txt2img.sh — useful for forcing the edit away
#                    from specific failure modes (e.g. "blurry,
#                    distorted, deformed, text, watermark").
#   EXTRA_IMAGES     comma-separated paths to ADDITIONAL reference
#                    images.  The first positional arg is still the
#                    primary image being edited; references in
#                    ``EXTRA_IMAGES`` are visual context the
#                    Qwen-Image-Edit-2511 model uses to condition
#                    the edit (e.g. "blend the subject of <primary>
#                    with the style of <ref1>" or "make <primary>
#                    look like <ref1> + <ref2>").  Up to 16
#                    references per the OpenAI image-edits spec.
#                    Each path encoded as base64 and sent as part
#                    of the polymorphic ``image: [primary, ref1, ...]``
#                    request field.
#   CFG_SCALE        prompt-faithfulness (default per yaml: 4.0).
#                    Higher = more aggressive edit toward the prompt.
#   STEPS            diffusion steps (default per yaml: 50).
#   SAMPLER          sampler (default per yaml: dpm++_2m).
#   SEED             integer seed (default -1 = random).
#
# Per-request body fields override the yaml defaults; leave a knob
# unset to inherit the global config.

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
BODY_FILE=$(mktemp -t img2img_body.XXXXXX)
TMP_FILES=("$B64_FILE" "$BODY_FILE")
cleanup () {
    rm -f "${TMP_FILES[@]}"
}
trap cleanup EXIT

encode_b64 () {
    local src="$1" dest="$2"
    if base64 --help 2>&1 | grep -q -- '-w'; then
        base64 -w0 "$src" > "$dest"
    else
        base64 < "$src" | tr -d '\n' > "$dest"
    fi
}

encode_b64 "$INPUT" "$B64_FILE"

# Build the body incrementally so unset env vars stay absent from the
# JSON (api treats absent fields as "use yaml default").
#
# ``image`` is polymorphic per the OpenAI image-edit spec: either a
# single string (just the primary image) or a list of strings (primary
# first, additional reference images after).  When EXTRA_IMAGES is
# set, we emit the list form so Qwen-Image-Edit-2511 sees the extras
# as conditioning context.  Same body field, two shapes.
JQ_ARGS=(
    --arg prompt "$PROMPT"
    --arg model "$MODEL"
    --argjson denoise "$DENOISE"
)
JQ_EXPR_PREFIX='{prompt: $prompt, model: $model, denoising_strength: $denoise'

if [[ -n "${EXTRA_IMAGES:-}" ]]; then
    # Comma-separated list of additional image paths.  Encode each
    # to base64 into its own temp file (so we can use --rawfile and
    # avoid argv bloat for multi-MB blobs).
    IMG_ARGS=(--rawfile _img0 "$B64_FILE")
    IMG_JQ='[$_img0'
    i=0
    IFS=',' read -ra EXTRA_PATHS <<< "$EXTRA_IMAGES"
    for path in "${EXTRA_PATHS[@]}"; do
        path="${path#"${path%%[![:space:]]*}"}"  # ltrim
        path="${path%"${path##*[![:space:]]}"}"  # rtrim
        [[ -z "$path" ]] && continue
        if [[ ! -f "$path" ]]; then
            echo "✘ EXTRA_IMAGES entry not found: $path" >&2
            exit 1
        fi
        i=$((i + 1))
        extra_b64=$(mktemp -t img2img_extra_b64.XXXXXX)
        TMP_FILES+=("$extra_b64")
        encode_b64 "$path" "$extra_b64"
        IMG_ARGS+=(--rawfile "_img$i" "$extra_b64")
        IMG_JQ+=", \$_img$i"
    done
    IMG_JQ+="]"
    JQ_ARGS+=("${IMG_ARGS[@]}")
    JQ_EXPR="${JQ_EXPR_PREFIX}, image: ${IMG_JQ}}"
else
    JQ_ARGS+=(--rawfile image "$B64_FILE")
    JQ_EXPR="${JQ_EXPR_PREFIX}, image: \$image}"
fi

add_str_field () {
    local field="$1" value="${2:-}"
    if [[ -n "$value" ]]; then
        JQ_ARGS+=(--arg "$field" "$value")
        JQ_EXPR="${JQ_EXPR%\}}, $field: \$$field}"
    fi
}
add_num_field () {
    local field="$1" value="${2:-}"
    if [[ -n "$value" ]]; then
        JQ_ARGS+=(--argjson "$field" "$value")
        JQ_EXPR="${JQ_EXPR%\}}, $field: \$$field}"
    fi
}

add_str_field negative_prompt "${NEGATIVE_PROMPT:-}"
add_str_field sampler_name    "${SAMPLER:-}"
add_num_field cfg_scale       "${CFG_SCALE:-}"
add_num_field steps           "${STEPS:-}"
add_num_field seed            "${SEED:-}"

jq -n "${JQ_ARGS[@]}" "$JQ_EXPR" > "$BODY_FILE"

echo "→ POST $API_BASE/v1/images/edits"
echo "  input              = $INPUT ($(wc -c < "$INPUT") bytes)"
echo "  prompt             = $PROMPT"
echo "  model              = $MODEL"
echo "  denoising_strength = $DENOISE"
[[ -n "${EXTRA_IMAGES:-}"    ]] && echo "  extra_images       = $EXTRA_IMAGES"
[[ -n "${NEGATIVE_PROMPT:-}" ]] && echo "  negative           = $NEGATIVE_PROMPT"
[[ -n "${CFG_SCALE:-}"       ]] && echo "  cfg_scale          = $CFG_SCALE"
[[ -n "${STEPS:-}"           ]] && echo "  steps              = $STEPS"
[[ -n "${SAMPLER:-}"         ]] && echo "  sampler            = $SAMPLER"
[[ -n "${SEED:-}"            ]] && echo "  seed               = $SEED"

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
