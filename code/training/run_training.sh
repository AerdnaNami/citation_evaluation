#!/usr/bin/env bash
set -euo pipefail

CONFIG="config_xlmr.json"
TRAIN_SCRIPT="train_xlmr_ner.py"

# Read category names from config
mapfile -t CATS < <(jq -r '.categories[].name' "$CONFIG" | sed '/^$/d')

for CAT in "${CATS[@]}"; do
  echo "========================================="
  echo "Training NER model for: $CAT"
  echo "========================================="

  python "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --category "$CAT"

done

echo "All categories finished."
