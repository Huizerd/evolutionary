#! /usr/bin/env bash

# Combine and analyze results from train_cases_7-8_left.sh and train_cases_7-8_right.sh
# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# Case 7
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/slowNsmall/slowNsmall"
cp "${LOGDIR}/slowNsmall+left+0/config.yaml" "${LOGDIR}/analysis/slowNsmall/config.yaml"
cp -a "${LOGDIR}/slowNsmall+left+0/hof_499" "${LOGDIR}/analysis/slowNsmall/slowNsmall/hof+left+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/slowNsmall/config.yaml" \
--parameters "${LOGDIR}/analysis/slowNsmall/slowNsmall" &
