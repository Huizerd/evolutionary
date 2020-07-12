#! /usr/bin/env bash

# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/runs
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# ANN + SSE D = 1
mkdir -p "${LOGDIR}/analysis/ann+sseD1/ann+sseD1"
cp "${LOGDIR}/ann+sseD1+0/config.yaml" "${LOGDIR}/analysis/ann+sseD1/config.yaml"
cp -a "${LOGDIR}/ann+sseD1+0/hof_199" "${LOGDIR}/analysis/ann+sseD1/ann+sseD1/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/ann+sseD1/config.yaml" \
--parameters "${LOGDIR}/analysis/ann+sseD1/ann+sseD1" &

# ANN + SSE D = 0.5
mkdir -p "${LOGDIR}/analysis/ann+sseD05/ann+sseD05"
cp "${LOGDIR}/ann+sseD05+0/config.yaml" "${LOGDIR}/analysis/ann+sseD05/config.yaml"
cp -a "${LOGDIR}/ann+sseD05+0/hof_199" "${LOGDIR}/analysis/ann+sseD05/ann+sseD05/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/ann+sseD05/config.yaml" \
--parameters "${LOGDIR}/analysis/ann+sseD05/ann+sseD05" &

# SNN + extra noise
mkdir -p "${LOGDIR}/analysis/snn+noisy/snn+noisy"
cp "${LOGDIR}/snn+noisy+0/config.yaml" "${LOGDIR}/analysis/snn+noisy/config.yaml"
cp -a "${LOGDIR}/snn+noisy+0/hof_199" "${LOGDIR}/analysis/snn+noisy/snn+noisy/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+noisy/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+noisy/snn+noisy" &

# SNN + SSE D = 1
mkdir -p "${LOGDIR}/analysis/snn+sseD1/snn+sseD1"
cp "${LOGDIR}/snn+sseD1+0/config.yaml" "${LOGDIR}/analysis/snn+sseD1/config.yaml"
cp -a "${LOGDIR}/snn+sseD1+0/hof_199" "${LOGDIR}/analysis/snn+sseD1/snn+sseD1/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD1/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD1/snn+sseD1" &

# SNN + SSE D = 0.5
mkdir -p "${LOGDIR}/analysis/snn+sseD05/snn+sseD05"
cp "${LOGDIR}/snn+sseD05+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+0/hof_199" "${LOGDIR}/analysis/snn+sseD05/snn+sseD05/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05/snn+sseD05" &

# SNN + SSE D = 0.5 + hdot
mkdir -p "${LOGDIR}/analysis/snn+sseD05+hdot/snn+sseD05+hdot"
cp "${LOGDIR}/snn+sseD05+hdot+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+hdot/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+hdot+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+hdot/snn+sseD05+hdot/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+hdot/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+hdot/snn+sseD05+hdot" &
