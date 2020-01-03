#! /usr/bin/env bash

# Combine and analyze results from train_cases_0-1_left.sh and train_cases_0-1_right.sh
# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# Case 0
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case0/case0"
cp "${LOGDIR}/case0+left+0/config.yaml" "${LOGDIR}/analysis/case0/config.yaml"
cp -a "${LOGDIR}/case0+left+0/hof_399" "${LOGDIR}/analysis/case0/case0/hof+left+0"
cp -a "${LOGDIR}/case0+left+1/hof_399" "${LOGDIR}/analysis/case0/case0/hof+left+1"
cp -a "${LOGDIR}/case0+right+0/hof_399" "${LOGDIR}/analysis/case0/case0/hof+right+0"
cp -a "${LOGDIR}/case0+right+1/hof_399" "${LOGDIR}/analysis/case0/case0/hof+right+1"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case0/config.yaml" \
--parameters "${LOGDIR}/analysis/case0/case0" &

# Case 1
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case1/case1"
cp "${LOGDIR}/case1+left+0/config.yaml" "${LOGDIR}/analysis/case1/config.yaml"
cp -a "${LOGDIR}/case1+left+0/hof_399" "${LOGDIR}/analysis/case1/case1/hof+left+0"
cp -a "${LOGDIR}/case1+left+1/hof_399" "${LOGDIR}/analysis/case1/case1/hof+left+1"
cp -a "${LOGDIR}/case1+right+0/hof_399" "${LOGDIR}/analysis/case1/case1/hof+right+0"
cp -a "${LOGDIR}/case1+right+1/hof_399" "${LOGDIR}/analysis/case1/case1/hof+right+1"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case1/config.yaml" \
--parameters "${LOGDIR}/analysis/case1/case1" &
