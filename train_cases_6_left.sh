#! /usr/bin/env bash

# Left: run parallel with right (since we only use half of the cores)
# Case 6
python main.py --config configs/case6a.yaml --tags case6a left
python main.py --config configs/case6b.yaml --tags case6b left
python main.py --config configs/case6c.yaml --tags case6c left
python main.py --config configs/case6d.yaml --tags case6d left
