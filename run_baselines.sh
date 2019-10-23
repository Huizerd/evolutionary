#!/usr/bin/env bash

python main.py --noplot --config configs/snn+landing1a.yaml --tags @snn @landing
python main.py --noplot --config configs/snn+landing1b.yaml --tags @snn @landing @8
python main.py --noplot --config configs/snn+landing1c.yaml --tags @snn @landing @4
python main.py --noplot --config configs/snn+landing2a.yaml --tags @snn @landing @inc
python main.py --noplot --config configs/snn+landing3a.yaml --tags @snn @landing @1in
python main.py --noplot --config configs/snn+landing3b.yaml --tags @snn @landing @1in1out
python main.py --noplot --config configs/snn+landing4a.yaml --tags @snn @landing @weighted
