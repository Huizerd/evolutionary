#! /usr/bin/env bash

# Combine and analyze results from train_cases_3-5_left.sh and train_cases_3-5_right.sh
# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

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

# Case 4a
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case4a/case4a"
cp "${LOGDIR}/case4a+left+0/config.yaml" "${LOGDIR}/analysis/case4a/config.yaml"
cp -a "${LOGDIR}/case4a+left+0/hof_399" "${LOGDIR}/analysis/case4a/case4a/hof+left+0"
cp -a "${LOGDIR}/case4a+right+0/hof_399" "${LOGDIR}/analysis/case4a/case4a/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case4a/config.yaml" \
--parameters "${LOGDIR}/analysis/case4a/case4a" &

# Case 4b
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case4b/case4b"
cp "${LOGDIR}/case4b+left+0/config.yaml" "${LOGDIR}/analysis/case4b/config.yaml"
cp -a "${LOGDIR}/case4b+left+0/hof_399" "${LOGDIR}/analysis/case4b/case4b/hof+left+0"
cp -a "${LOGDIR}/case4b+right+0/hof_399" "${LOGDIR}/analysis/case4b/case4b/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case4b/config.yaml" \
--parameters "${LOGDIR}/analysis/case4b/case4b" &

# Case 4c
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case4c/case4c"
cp "${LOGDIR}/case4c+left+0/config.yaml" "${LOGDIR}/analysis/case4c/config.yaml"
cp -a "${LOGDIR}/case4c+left+0/hof_399" "${LOGDIR}/analysis/case4c/case4c/hof+left+0"
cp -a "${LOGDIR}/case4c+right+0/hof_399" "${LOGDIR}/analysis/case4c/case4c/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case4c/config.yaml" \
--parameters "${LOGDIR}/analysis/case4c/case4c" &

# Case 5a
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case5a/case5a"
cp "${LOGDIR}/case5a+left+0/config.yaml" "${LOGDIR}/analysis/case5a/config.yaml"
cp -a "${LOGDIR}/case5a+left+0/hof_399" "${LOGDIR}/analysis/case5a/case5a/hof+left+0"
cp -a "${LOGDIR}/case5a+right+0/hof_399" "${LOGDIR}/analysis/case5a/case5a/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case5a/config.yaml" \
--parameters "${LOGDIR}/analysis/case5a/case5a" &

# Case 5b
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case5b/case5b"
cp "${LOGDIR}/case5b+left+0/config.yaml" "${LOGDIR}/analysis/case5b/config.yaml"
cp -a "${LOGDIR}/case5b+left+0/hof_399" "${LOGDIR}/analysis/case5b/case5b/hof+left+0"
cp -a "${LOGDIR}/case5b+right+0/hof_399" "${LOGDIR}/analysis/case5b/case5b/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case5b/config.yaml" \
--parameters "${LOGDIR}/analysis/case5b/case5b" &

# Case 5c
# Create folder and copy data
mkdir -p "${LOGDIR}/analysis/case5c/case5c"
cp "${LOGDIR}/case5c+left+0/config.yaml" "${LOGDIR}/analysis/case5c/config.yaml"
cp -a "${LOGDIR}/case5c+left+0/hof_399" "${LOGDIR}/analysis/case5c/case5c/hof+left+0"
cp -a "${LOGDIR}/case5c+right+0/hof_399" "${LOGDIR}/analysis/case5c/case5c/hof+right+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/case5c/config.yaml" \
--parameters "${LOGDIR}/analysis/case5c/case5c" &
