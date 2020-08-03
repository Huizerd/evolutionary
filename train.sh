#! /usr/bin/env bash

python main.py --config configs/defaults.yaml --tags defaults Dblind
python main.py --config configs/defaults.yaml --tags defaults Dblind

python main.py --config configs/2x5neurons.yaml --tags 2x5neurons Dblind
python main.py --config configs/2x5neurons.yaml --tags 2x5neurons Dblind
