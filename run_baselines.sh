#!/usr/bin/env bash

python main.py --noplot --config configs/snn+landing.yaml --tags @snn @landing
python main.py --noplot --config configs/ann+landing.yaml --tags @ann @landing
