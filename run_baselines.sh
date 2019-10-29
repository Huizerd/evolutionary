#!/usr/bin/env bash

python main.py --verbose 1 --config configs/snn+landing1.yaml --tags @standard
python main.py --verbose 1 --config configs/snn+landing2.yaml --tags @incremental2
