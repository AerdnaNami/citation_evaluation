#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh train.py config.json
#
# This script will run the same TRAIN_SCRIPT twice:
#   1) with config.json
#   2) then with config_scilitllm.json (if it exists)

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <train_script.py> <config.json>"
  exit 1
fi

TRAIN_SCRIPT="$1"
CONFIG1="$2"
CONFIG2="config_scilitllm.json"

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Train script not found: $TRAIN_SCRIPT"
  exit 1
fi

if [ ! -f "$CONFIG1" ]; then
  echo "Config file not found: $CONFIG1"
  exit 1
fi

run_config () {
  local CONFIG="$1"

  if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
  fi

  mapfile -t CATS < <(jq -r '.categories[].name' "$CONFIG" | sed '/^$/d')
  
  for CAT in "${CATS[@]}"
  do 
  echo "Training ($CONFIG): $CAT"
  python "$TRAIN_SCRIPT" --config "$CONFIG" --category "$CAT"
  done

}

# Run the provided config first
run_config "$CONFIG1"

# Then run config_scilitllm.json
run_config "$CONFIG2"
