#! /usr/bin/env bash

# Right: run parallel with left (since we only use half of the cores)
# Case 6
python main.py --config configs/case6a.yaml --tags case6a right
python main.py --config configs/case6b.yaml --tags case6b right
python main.py --config configs/case6c.yaml --tags case6c right
python main.py --config configs/case6d.yaml --tags case6d right
