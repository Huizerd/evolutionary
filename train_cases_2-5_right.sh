#! /usr/bin/env bash

# Right: run parallel with left (since we only use half of the cores)
# Case 2
python main.py --config configs/case2a.yaml --tags case2a right
python main.py --config configs/case2b.yaml --tags case2b right

# Case 3
python main.py --config configs/case3a.yaml --tags case3a right
python main.py --config configs/case3b.yaml --tags case3b right

# Case 4
python main.py --config configs/case4a.yaml --tags case4a right
python main.py --config configs/case4b.yaml --tags case4b right
python main.py --config configs/case4c.yaml --tags case4c right

# Case 5
python main.py --config configs/case5a.yaml --tags case5a right
python main.py --config configs/case5b.yaml --tags case5b right
python main.py --config configs/case5c.yaml --tags case5c right
