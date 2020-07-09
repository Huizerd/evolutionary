#! /usr/bin/env bash

# Check if cases directory exists (and if in correct folder)
LOGDIR=logs/cases
if [ -d "$LOGDIR" ]; then
  echo "Starting analysis"
else
  echo "Run script from evolutionary root!"
  exit 1
fi

# SNN + SSE D = 0.5 + no splitting of observations + weights (-1, 1) for 400 gen
mkdir -p "${LOGDIR}/analysis/snn+sseD05+nosplit+symweights+400/snn+sseD05+nosplit+symweights+400"
cp "${LOGDIR}/snn+sseD05+nosplit+symweights+400+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+nosplit+symweights+400/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+nosplit+symweights+400+0/hof_399" "${LOGDIR}/analysis/snn+sseD05+nosplit+symweights+400/snn+sseD05+nosplit+symweights+400/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+nosplit+symweights+400/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+nosplit+symweights+400/snn+sseD05+nosplit+symweights+400" &

# SNN + SSE D = 0.5 + no splitting of observations + no clamping of alpha/tau + weights (-1, 1)
mkdir -p "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+symweights/snn+sseD05+nosplit+noclamp+symweights"
cp "${LOGDIR}/snn+sseD05+nosplit+noclamp+symweights+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+symweights/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+nosplit+noclamp+symweights+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+symweights/snn+sseD05+nosplit+noclamp+symweights/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+symweights/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+symweights/snn+sseD05+nosplit+noclamp+symweights" &

# SNN + SSE D = 0.5 + no splitting of observations + non-spiking decoding + weights (-1, 1)
mkdir -p "${LOGDIR}/analysis/snn+sseD05+nosplit+nospike+symweights/snn+sseD05+nosplit+nospike+symweights"
cp "${LOGDIR}/snn+sseD05+nosplit+nospike+symweights+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+nosplit+nospike+symweights/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+nosplit+nospike+symweights+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+nosplit+nospike+symweights/snn+sseD05+nosplit+nospike+symweights/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+nosplit+nospike+symweights/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+nosplit+nospike+symweights/snn+sseD05+nosplit+nospike+symweights" &

# SNN + SSE D = 0.5 + no splitting of observations + no clamping of alpha/tau + non-spiking decoding + weights (-1, 1)
mkdir -p "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+symweights/snn+sseD05+nosplit+noclamp+nospike+symweights"
cp "${LOGDIR}/snn+sseD05+nosplit+noclamp+nospike+symweights+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+symweights/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+nosplit+noclamp+nospike+symweights+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+symweights/snn+sseD05+nosplit+noclamp+nospike+symweights/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+symweights/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+symweights/snn+sseD05+nosplit+noclamp+nospike+symweights" &

# SNN + SSE D = 0.5 + no splitting of observations + no clamping of alpha/tau + non-spiking decoding + output between -7.848 and 4.905 + weights (-1, 1)
mkdir -p "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights"
cp "${LOGDIR}/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights+0/config.yaml" "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights/config.yaml"
cp -a "${LOGDIR}/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights+0/hof_199" "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights/hof+0"
# Run analysis in background (since it only uses 1 core)
python main.py \
--mode analyze4m \
--config "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights/config.yaml" \
--parameters "${LOGDIR}/analysis/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights/snn+sseD05+nosplit+noclamp+nospike+wideout+symweights" &

