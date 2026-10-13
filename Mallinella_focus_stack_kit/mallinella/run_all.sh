#!/usr/bin/env bash
# Render every specimen in specimens/ with the views of a config file.
#   ./run_all.sh config_example.json            (BLENDER=/path/to/blender ./run_all.sh ... to override)
#   N_NEW=20 ./run_all.sh config_example.json   also generates 20 new random specimens first
set -e
CONFIG=${1:-config_example.json}
BLENDER=${BLENDER:-blender}
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$N_NEW" ]; then
  START=${SEED_START:-100}
  for ((s=START; s<START+N_NEW; s++)); do
    "$BLENDER" -b -P "$HERE/mallinella_generator.py" -- --preset random --seed $s \
      --out "$HERE/specimens/mallinella_random_$(printf %03d $s).blend"
  done
fi
for f in "$HERE"/specimens/*.blend; do
  "$BLENDER" -b "$f" -P "$HERE/render_focus_stack.py" -- --config "$CONFIG"
done
