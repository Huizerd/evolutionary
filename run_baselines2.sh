#!/usr/bin/env bash

python main.py --noplot --config configs/snn+hover2.yaml --tags @nonothing @signed+offset @equaldynamics @doubleneuron
python main.py --noplot --config configs/snn+landing2.yaml --tags @landing @equaldynamics @doubleneuron
