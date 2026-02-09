#!/usr/bin/env bash
set -euo pipefail

CONFIG="config.json"
TRAIN_SCRIPT="train.py"

mapfile -t CATS < <(jq -r '.categories[].name' "$CONFIG" | sed '/^$/d')

for CAT in "${CATS[@]}"; do
  echo "Training: $CAT"
  python "$TRAIN_SCRIPT" --config "$CONFIG" --category "$CAT"
done
