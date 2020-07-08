#! /usr/bin/env bash

# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# SNN + SSE D = 0.5 + no splitting of observations
mkdir -p "${LOGDIR}/analysis/snn+sseD05+nosplit/snn+sseD05+nosplit"
cp "${LOGDIR}/snn+sseD05+nosplit+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+nosplit/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+nosplit+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+nosplit/snn+sseD05+nosplit/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+nosplit/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+nosplit/snn+sseD05+nosplit" &

# SNN + SSE D = 0.5 + don't go above starting h
mkdir -p "${LOGDIR}/analysis/snn+sseD05+hnogo/snn+sseD05+hnogo"
cp "${LOGDIR}/snn+sseD05+hnogo+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+hnogo/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+hnogo+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+hnogo/snn+sseD05+hnogo/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+hnogo/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+hnogo/snn+sseD05+hnogo" &
