#! /usr/bin/env bash

python main.py --config configs/snn+sseD05+nosplit.yaml --tags snn sseD05 nosplit symweights 400
python main.py --config configs/snn+sseD05+nosplit+noclamp.yaml --tags snn sseD05 nosplit noclamp symweights
python main.py --config configs/snn+sseD05+nosplit+nospike.yaml --tags snn sseD05 nosplit nospike symweights
python main.py --config configs/snn+sseD05+nosplit+noclamp+nospike.yaml --tags snn sseD05 nosplit noclamp nospike symweights
python main.py --config configs/snn+sseD05+nosplit+noclamp+nospike+wideout.yaml --tags snn sseD05 nosplit noclamp nospike wideout symweights
