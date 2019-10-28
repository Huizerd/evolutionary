#!/usr/bin/env bash

python main.py --verbose 1 --config configs/snn+landing.yaml --tags @baseline
python main.py --verbose 1 --config configs/snn+landing1a.yaml --tags @mutpb0.7
python main.py --verbose 1 --config configs/snn+landing1b.yaml --tags @incremental
python main.py --verbose 1 --config configs/snn+landing1c.yaml --tags @incremental2
