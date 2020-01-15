#! /usr/bin/env bash

# Combine and analyze results from train_cases_6_left.sh and train_cases_6_right.sh
# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# Case 6a
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case6a/case6a"
cp "${LOGDIR}/case6a+left+0/config.yaml" "${LOGDIR}/analysis/case6a/config.yaml"
cp -a "${LOGDIR}/case6a+left+0/hof_399" "${LOGDIR}/analysis/case6a/case6a/hof+left+0"
cp -a "${LOGDIR}/case6a+right+0/hof_399" "${LOGDIR}/analysis/case6a/case6a/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case6a/config.yaml" \
--parameters "${LOGDIR}/analysis/case6a/case6a" &

# Case 6b
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case6b/case6b"
cp "${LOGDIR}/case6b+left+0/config.yaml" "${LOGDIR}/analysis/case6b/config.yaml"
cp -a "${LOGDIR}/case6b+left+0/hof_399" "${LOGDIR}/analysis/case6b/case6b/hof+left+0"
cp -a "${LOGDIR}/case6b+right+0/hof_399" "${LOGDIR}/analysis/case6b/case6b/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case6b/config.yaml" \
--parameters "${LOGDIR}/analysis/case6b/case6b" &

# Case 6c
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case6c/case6c"
cp "${LOGDIR}/case6c+left+0/config.yaml" "${LOGDIR}/analysis/case6c/config.yaml"
cp -a "${LOGDIR}/case6c+left+0/hof_399" "${LOGDIR}/analysis/case6c/case6c/hof+left+0"
cp -a "${LOGDIR}/case6c+right+0/hof_399" "${LOGDIR}/analysis/case6c/case6c/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case6c/config.yaml" \
--parameters "${LOGDIR}/analysis/case6c/case6c" &

# Case 6d
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case6d/case6d"
cp "${LOGDIR}/case6d+left+0/config.yaml" "${LOGDIR}/analysis/case6d/config.yaml"
cp -a "${LOGDIR}/case6d+left+0/hof_399" "${LOGDIR}/analysis/case6d/case6d/hof+left+0"
cp -a "${LOGDIR}/case6d+right+0/hof_399" "${LOGDIR}/analysis/case6d/case6d/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case6d/config.yaml" \
--parameters "${LOGDIR}/analysis/case6d/case6d" &
