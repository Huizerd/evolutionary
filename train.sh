#! /usr/bin/env bash

python main.py --config configs/ann+sseD1.yaml --tags ann sseD1
python main.py --config configs/ann+sseD05.yaml --tags ann sseD05
python main.py --config configs/snn+noisy.yaml --tags snn noisy
python main.py --config configs/snn+sseD1.yaml --tags snn sseD1
python main.py --config configs/snn+sseD05.yaml --tags snn sseD05
python main.py --config configs/snn+sseD05+hdot.yaml --tags snn sseD05 hdot
