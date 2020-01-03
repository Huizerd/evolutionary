#! /usr/bin/env bash

# Left: run parallel with right (since we only use half of the cores)
# Case 2
python main.py --config configs/case2a.yaml --tags case2a left
python main.py --config configs/case2b.yaml --tags case2b left

# Case 3
python main.py --config configs/case3a.yaml --tags case3a left
python main.py --config configs/case3b.yaml --tags case3b left

# Case 4
python main.py --config configs/case4a.yaml --tags case4a left
python main.py --config configs/case4b.yaml --tags case4b left
python main.py --config configs/case4c.yaml --tags case4c left

# Case 5
python main.py --config configs/case5a.yaml --tags case5a left
python main.py --config configs/case5b.yaml --tags case5b left
python main.py --config configs/case5c.yaml --tags case5c left
