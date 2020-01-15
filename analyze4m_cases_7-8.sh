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
mkdir -p "${LOGDIR}/analysis/case7/case7"
cp "${LOGDIR}/case7+left+0/config.yaml" "${LOGDIR}/analysis/case7/config.yaml"
cp -a "${LOGDIR}/case7+left+0/hof_399" "${LOGDIR}/analysis/case7/case7/hof+left+0"
cp -a "${LOGDIR}/case7+left+1/hof_399" "${LOGDIR}/analysis/case7/case7/hof+left+1"
cp -a "${LOGDIR}/case7+right+0/hof_399" "${LOGDIR}/analysis/case7/case7/hof+right+0"
cp -a "${LOGDIR}/case7+right+1/hof_399" "${LOGDIR}/analysis/case7/case7/hof+right+1"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case7/config.yaml" \
--parameters "${LOGDIR}/analysis/case7/case7" &

# Case 8
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case8/case8"
cp "${LOGDIR}/case8+left+0/config.yaml" "${LOGDIR}/analysis/case8/config.yaml"
cp -a "${LOGDIR}/case8+left+0/hof_399" "${LOGDIR}/analysis/case8/case8/hof+left+0"
cp -a "${LOGDIR}/case8+left+1/hof_399" "${LOGDIR}/analysis/case8/case8/hof+left+1"
cp -a "${LOGDIR}/case8+right+0/hof_399" "${LOGDIR}/analysis/case8/case8/hof+right+0"
cp -a "${LOGDIR}/case8+right+1/hof_399" "${LOGDIR}/analysis/case8/case8/hof+right+1"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case8/config.yaml" \
--parameters "${LOGDIR}/analysis/case8/case8" &
