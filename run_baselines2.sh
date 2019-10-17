#!/usr/bin/env bash

python main.py --noplot --config configs/snn+hover3.yaml --tags @nonothing @dummy+offset5m @equaldynamics
python main.py --noplot --config configs/snn+hover1.yaml --tags @nonothing @unsigned+signed @equaldynamics
