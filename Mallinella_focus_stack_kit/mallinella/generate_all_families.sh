#!/usr/bin/env bash
set -euo pipefail

# Generate every declared family morphotype and render every configured view.
# Usage:
#   ./generate_all_families.sh [config.json] [--quick]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-"$HERE/config_example.json"}"
QUICK=0
if [[ "${2:-}" == "--quick" || "${1:-}" == "--quick" ]]; then
    QUICK=1
    CONFIG="${1:-"$HERE/config_example.json"}"
    [[ "$CONFIG" == "--quick" ]] && CONFIG="$HERE/config_example.json"
fi

BLENDER="${BLENDER:-blender}"
SEED="${SEED:-42}"
HAIR="${HAIR:-1.0}"
SPECIMENS="$HERE/specimens"
GENERATOR="$HERE/mallinella_generator.py"
RENDERER="$HERE/render_focus_stack.py"

PRESETS=(
    sphaerica_f
    sphaerica_m
    pluma_m
    ctenus_generic
    macroctenus_generic
    amicactenus_generic
)

for preset in "${PRESETS[@]}"; do
    blend="$SPECIMENS/${preset}_${SEED}.blend"

    "$BLENDER" -b -P "$GENERATOR" -- \
        --preset "$preset" \
        --seed "$SEED" \
        --hair "$HAIR" \
        --out "$blend"

    render_args=(
        -b "$blend"
        -P "$RENDERER" --
        --config "$CONFIG"
    )
    if (( QUICK )); then
        render_args+=(--quick)
    fi
    "$BLENDER" "${render_args[@]}"
done
