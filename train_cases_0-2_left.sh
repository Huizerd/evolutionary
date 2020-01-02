#! /usr/bin/env bash

# Left: run parallel with right (since we only use half of the cores)
# Case 0
python main.py --config configs/case0.yaml --tags case0 left
python main.py --config configs/case0.yaml --tags case0 left

# Case 1
python main.py --config configs/case1.yaml --tags case1 left
python main.py --config configs/case1.yaml --tags case1 left

# Case 2
python main.py --config configs/case2.yaml --tags case2 left
python main.py --config configs/case2.yaml --tags case2 left
