#! /usr/bin/env bash

# Right: run parallel with left (since we only use half of the cores)
# Case 7
python main.py --config configs/case7.yaml --tags case7 right
python main.py --config configs/case7.yaml --tags case7 right

# Case 8
python main.py --config configs/case8.yaml --tags case8 right
python main.py --config configs/case8.yaml --tags case8 right
