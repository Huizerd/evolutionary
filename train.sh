#! /usr/bin/env bash

python main.py --config configs/final.yaml --tags float16-enc quantized
python main.py --config configs/final.yaml --tags float16-enc quantized
