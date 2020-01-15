#! /usr/bin/env bash

# Left: run parallel with right (since we only use half of the cores)
# Case 7
python main.py --config configs/case7.yaml --tags case7 left
python main.py --config configs/case7.yaml --tags case7 left

# Case 8
python main.py --config configs/case8.yaml --tags case8 left
python main.py --config configs/case8.yaml --tags case8 left
