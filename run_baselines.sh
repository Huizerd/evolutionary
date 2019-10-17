#!/usr/bin/env bash

python main.py --noplot --config configs/snn+hover.yaml --tags @nonothing @signed+offset @equaldynamics
python main.py --noplot --config configs/snn+hover2.yaml --tags @nonothing @signed+offset @equaldynamics @doubleneuron
python main.py --noplot --config configs/snn+landing.yaml --tags @landing @equaldynamics
python main.py --noplot --config configs/snn+landing2.yaml --tags @landing @equaldynamics @doubleneuron
python main.py --noplot --config configs/ann+hover.yaml --tags @nonothing @signed+offset
python main.py --noplot --config configs/ann+landing.yaml --tags @landing
