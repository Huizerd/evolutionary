#!/usr/bin/env bash

python main.py --noplot --config configs/snn+landing.yaml --tags @snn @landing
python main.py --noplot --config configs/snn+landing2.yaml --tags @snn @landing @8hidden
python main.py --noplot --config configs/snn+landing3.yaml --tags @snn @landing @weights
