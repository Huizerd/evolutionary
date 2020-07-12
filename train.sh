#! /usr/bin/env bash

# This is about trying to find out what is impeding learning in SNNs
# - SNN with 5 hidden neurons can land after 20 (!) generations already
# - SNN with 20 hidden neurons didn't even learn to land after 200 generations
#
# So, we try multiple things:
# - reproduce the learning multiple times for SNN with 5 hidden (100 generations) -> snn+5neurons+100.yaml
# - check the impediment to learning with more hidden neurons is only the search space, i.e., more generations should work (1000 generations) -> snn+20neurons+1000.yaml
#
# Next, there can be other causes, such as there being multiple (reachable) optima in the search space for SNNs. Causes of this could be:
# - the fact that long, okayish landings have a higher sum of squared D errors than quick out-of-bounds by going up
# - the fact that SNNs seem to lack the fine resolution of ANNs, whose landings are inherently more optimal, and therefore never end up ascending
#
# To deal with these, we test some more:
# - the effect of having the average of squared D errors as objective with 20 hidden neurons (200 generations) -> snn+20neurons+avgD.yaml
# - the effect of increasing the bound on altitude from h0 + 5 to 100 (200 generations) -> snn+20neurons+high.yaml

python main.py --config configs/snn+5neurons+100.yaml --tags snn 5neurons
python main.py --config configs/snn+5neurons+100.yaml --tags snn 5neurons
python main.py --config configs/snn+5neurons+100.yaml --tags snn 5neurons

python main.py --config configs/snn+20neurons+1000.yaml --tags snn 20neurons
python main.py --config configs/snn+20neurons+1000.yaml --tags snn 20neurons

python main.py --config configs/snn+20neurons+avgD.yaml --tags snn 20neurons avgD
python main.py --config configs/snn+20neurons+avgD.yaml --tags snn 20neurons avgD
python main.py --config configs/snn+20neurons+avgD.yaml --tags snn 20neurons avgD

python main.py --config configs/snn+20neurons+high.yaml --tags snn 20neurons high
python main.py --config configs/snn+20neurons+high.yaml --tags snn 20neurons high
python main.py --config configs/snn+20neurons+high.yaml --tags snn 20neurons high
