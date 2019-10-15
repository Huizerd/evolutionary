#!/usr/bin/env bash

python main.py --config configs/snn+hover2.yaml --tags @nonothing @unsigned+offset @equaldynamics @only5m
python main.py --config configs/snn+hover3.yaml --tags @nonothing @finalvel+offset5m @equaldynamics
python main.py --config configs/snn+hover4.yaml --tags @nonothing @signed+offset @equaldynamics
python main.py --config configs/snn+hover5.yaml --tags @nonothing @signed+offset @equaldynamics @weight+alpha+tau
python main.py --config configs/snn+hover6.yaml --tags @nonothing @signed+offset @equaldynamics @flipweights
