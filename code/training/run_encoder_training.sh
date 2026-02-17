#!/usr/bin/env bash
set -euo pipefail

# Two (config,train_script) pairs. Add more pairs here if needed.
CONFIGS=(
  "configs/config_xlmr.json"
  "configs/config_scibert.json"
)
TRAIN_SCRIPTS=(
  "train_xlmr_ner.py"
  "train_scibert.py"
)

# sanity: arrays must be same length
if [ "${#CONFIGS[@]}" -ne "${#TRAIN_SCRIPTS[@]}" ]; then
  echo "ERROR: CONFIGS and TRAIN_SCRIPTS arrays must have the same length."
  exit 1
fi

NUM_PAIRS=${#CONFIGS[@]}

for ((i=0; i<NUM_PAIRS; i++)); do
  CONFIG="${CONFIGS[i]}"
  TRAIN_SCRIPT="${TRAIN_SCRIPTS[i]}"

  if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
  fi

  if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Train script not found: $TRAIN_SCRIPT"
    exit 1
  fi

  echo
  echo "-----------------------------------------"
  echo "Running pair: "
  echo "  config: $CONFIG"
  echo "  train script: $TRAIN_SCRIPT"
  echo "-----------------------------------------"

  # Read category names from config (skip empty lines)
  mapfile -t CATS < <(jq -r '.categories[].name' "$CONFIG" | sed '/^$/d')

  if [ "${#CATS[@]}" -eq 0 ]; then
    echo "Warning: no categories found in $CONFIG (jq produced zero results). Skipping."
    continue
  fi

  for CAT in "${CATS[@]}"; do
    echo "========================================="
    echo "Training NER model for pair index $i, category: $CAT"
    echo "  using config: $CONFIG"
    echo "  using script: $TRAIN_SCRIPT"
    echo "========================================="

    python "$TRAIN_SCRIPT" \
      --config "$CONFIG" \
      --category "$CAT"

  done

done

echo
echo "All categories finished."
