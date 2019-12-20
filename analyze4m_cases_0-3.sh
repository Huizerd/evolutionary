#! /usr/bin/env bash

# Combine and analyze results from train_cases_0-3_left.sh and train_cases_0-3_right.sh
# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# Case 0
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case0/config.yaml" \
--parameters "${LOGDIR}/analysis/case0/case0" &

# Case 1
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case1/config.yaml" \
--parameters "${LOGDIR}/analysis/case1/case1" &

# Case 2
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case2/config.yaml" \
--parameters "${LOGDIR}/analysis/case2/case2" &

# Case 3
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case3/config.yaml" \
--parameters "${LOGDIR}/analysis/case3/case3" &
