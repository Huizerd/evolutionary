#! /usr/bin/env bash

# Right: run parallel with left (since we only use half of the cores)
# Case 0
python main.py --config configs/case0.yaml --tags case0 right
python main.py --config configs/case0.yaml --tags case0 right

# Case 1
python main.py --config configs/case1.yaml --tags case1 right
python main.py --config configs/case1.yaml --tags case1 right
