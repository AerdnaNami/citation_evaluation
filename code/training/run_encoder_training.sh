#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/config_xlmr.json"

if [ ! -f "$CONFIG" ]; then
  echo "Config file not found: $CONFIG"
  exit 1
fi

# Get categories as compact JSON objects
mapfile -t CAT_OBJS < <(jq -c '.categories[]' "$CONFIG")

if [ "${#CAT_OBJS[@]}" -eq 0 ]; then
  echo "Warning: no categories found in $CONFIG"
  exit 0
fi

for CAT_OBJ in "${CAT_OBJS[@]}"; do
  CAT_NAME=$(jq -r '.name' <<<"$CAT_OBJ")
  TRAIN_SCRIPT=$(jq -r '.train_script' <<<"$CAT_OBJ")

  if [ -z "$CAT_NAME" ] || [ "$CAT_NAME" = "null" ]; then
    echo "ERROR: category missing .name"
    exit 1
  fi

  if [ -z "$TRAIN_SCRIPT" ] || [ "$TRAIN_SCRIPT" = "null" ]; then
    echo "ERROR: category '$CAT_NAME' missing .train_script"
    exit 1
  fi

  if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Train script not found for category '$CAT_NAME': $TRAIN_SCRIPT"
    exit 1
  fi

  echo "========================================="
  echo "Training category: $CAT_NAME"
  echo "  config: $CONFIG"
  echo "  script: $TRAIN_SCRIPT"
  echo "========================================="

  python "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --category "$CAT_NAME"
done

echo
echo "All categories finished."
