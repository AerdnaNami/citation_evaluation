#!/usr/bin/env bash
set -euo pipefail

PAIRS=(
  "train_llama-3.1-8b-it.py configs/config_llama-3.1-8b-it.json"
  "train_llama-3.2-3b-it.py configs/config_llama-3.2-3b-it.json"
  "train_scitlitllm.py configs/config_scilitllm.json"
  "train_scitulu.py configs/config_scitulu.json"
)

run_pair () {
  local TRAIN_SCRIPT="$1"
  local CONFIG="$2"

  if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Train script not found: $TRAIN_SCRIPT"
    exit 1
  fi

  if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
  fi

  mapfile -t CATS < <(jq -r '.categories[].name' "$CONFIG" | sed '/^$/d')

  for CAT in "${CATS[@]}"; do
    echo "Training ($TRAIN_SCRIPT | $CONFIG): $CAT"
    python "$TRAIN_SCRIPT" --config "$CONFIG" --category "$CAT"
  done
}

for PAIR in "${PAIRS[@]}"; do
  # split "script config" into two vars
  read -r SCRIPT CFG <<< "$PAIR"
  run_pair "$SCRIPT" "$CFG"
done
