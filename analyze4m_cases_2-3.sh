#! /usr/bin/env bash

# Combine and analyze results from train_cases_2-5_left.sh and train_cases_2-5_right.sh
# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# Case 2a
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case2a/case2a"
cp "${LOGDIR}/case2a+left+0/config.yaml" "${LOGDIR}/analysis/case2a/config.yaml"
cp -a "${LOGDIR}/case2a+left+0/hof_399" "${LOGDIR}/analysis/case2a/case2a/hof+left+0"
cp -a "${LOGDIR}/case2a+right+0/hof_399" "${LOGDIR}/analysis/case2a/case2a/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case2a/config.yaml" \
--parameters "${LOGDIR}/analysis/case2a/case2a" &

# Case 2b
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case2b/case2b"
cp "${LOGDIR}/case2b+left+0/config.yaml" "${LOGDIR}/analysis/case2b/config.yaml"
cp -a "${LOGDIR}/case2b+left+0/hof_399" "${LOGDIR}/analysis/case2b/case2b/hof+left+0"
cp -a "${LOGDIR}/case2b+right+0/hof_399" "${LOGDIR}/analysis/case2b/case2b/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case2b/config.yaml" \
--parameters "${LOGDIR}/analysis/case2b/case2b" &

# Case 3a
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case3a/case3a"
cp "${LOGDIR}/case3a+left+0/config.yaml" "${LOGDIR}/analysis/case3a/config.yaml"
cp -a "${LOGDIR}/case3a+left+0/hof_399" "${LOGDIR}/analysis/case3a/case3a/hof+left+0"
cp -a "${LOGDIR}/case3a+right+0/hof_399" "${LOGDIR}/analysis/case3a/case3a/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case3a/config.yaml" \
--parameters "${LOGDIR}/analysis/case3a/case3a" &

# Case 3b
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case3b/case3b"
cp "${LOGDIR}/case3b+left+0/config.yaml" "${LOGDIR}/analysis/case3b/config.yaml"
cp -a "${LOGDIR}/case3b+left+0/hof_399" "${LOGDIR}/analysis/case3b/case3b/hof+left+0"
cp -a "${LOGDIR}/case3b+right+0/hof_399" "${LOGDIR}/analysis/case3b/case3b/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case3b/config.yaml" \
--parameters "${LOGDIR}/analysis/case3b/case3b" &
