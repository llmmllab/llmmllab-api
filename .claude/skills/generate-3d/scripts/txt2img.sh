#!/usr/bin/env bash
#
# txt2img.sh — exercise POST /v1/images/generations.
#
# Usage:
#   ./scripts/txt2img.sh "a teacup with steam"
#   ./scripts/txt2img.sh "a teacup with steam" qwen-image-2512 1024x1024
#
# Env overrides:
#   API_BASE         default http://localhost:8000
#   API_KEY          bearer token; omit for unauth dev endpoints
#   OUT_DIR          where to drop the decoded PNG (default ./out)
#   NEGATIVE_PROMPT  things to exclude from the image — strongly
#                    recommended for object-specific prompts.
#                    Examples:
#                      "blurry, distorted geometry, deformed, text, watermark"
#                      "G-clamp, vise, pliers" (when prompting for a C-clamp)
#   CFG_SCALE        prompt-faithfulness knob.  Defaults to the
#                    model's yaml value (currently 4.0 for qwen-image).
#                    Lower = more creative / aesthetic; higher = sticks
#                    closer to the prompt.  Usable range 1.5–8.
#   STEPS            number of diffusion steps (default per yaml: 50).
#                    More = finer details, longer wall-clock.
#   SAMPLER          sampler name.  Defaults to ``dpm++_2m`` (sharper
#                    geometry than euler; same speed).  Other options:
#                    ``euler``, ``dpm++_sde``, ``unipc``, ``dpmpp_2m_sde``.
#   SEED             integer seed for reproducible runs (default -1 = random).
#
# Per-request body fields override the model's yaml defaults, so any
# of CFG_SCALE / STEPS / SAMPLER / SEED you pass here wins over the
# global config.  Leave a knob unset to inherit the yaml default.

set -euo pipefail

PROMPT="${1:-a teacup with steam}"
MODEL="${2:-qwen-image-2512}"
SIZE="${3:-1024x1024}"

API_BASE="${API_BASE:-http://192.168.0.71:9999}"
OUT_DIR="${OUT_DIR:-./out}"
mkdir -p "$OUT_DIR"

API_KEY=$LLMMLL_AUTH_TOKEN

AUTH_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

TS=$(date +%s)
RESP_FILE="$OUT_DIR/txt2img_${TS}.json"
PNG_FILE="$OUT_DIR/txt2img_${TS}.png"

# Build the request body incrementally so unset env vars stay absent
# from the JSON (the api treats absent fields as "use yaml default",
# which is what we want).
JQ_ARGS=(
    --arg prompt "$PROMPT"
    --arg model "$MODEL"
    --arg size "$SIZE"
)
JQ_EXPR='{prompt: $prompt, model: $model, size: $size, n: 1}'

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

BODY=$(jq -n "${JQ_ARGS[@]}" "$JQ_EXPR")

echo "→ POST $API_BASE/v1/images/generations"
echo "  prompt = $PROMPT"
echo "  model  = $MODEL"
echo "  size   = $SIZE"
[[ -n "${NEGATIVE_PROMPT:-}" ]] && echo "  negative = $NEGATIVE_PROMPT"
[[ -n "${CFG_SCALE:-}"       ]] && echo "  cfg_scale = $CFG_SCALE"
[[ -n "${STEPS:-}"           ]] && echo "  steps     = $STEPS"
[[ -n "${SAMPLER:-}"         ]] && echo "  sampler   = $SAMPLER"
[[ -n "${SEED:-}"            ]] && echo "  seed      = $SEED"

curl -sS -X POST "$API_BASE/v1/images/generations" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
    -d "$BODY" \
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
