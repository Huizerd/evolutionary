import argparse
import os
import yaml

import torch
import pandas as pd

from evolutionary.network.snn import TwoLayerSNN, ThreeLayerSNN
from evolutionary.utils.constructors import build_network


def weights_to_csv(folder, parameters):
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder + f"/weights+{individual_id}/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load config
    with open(folder + "/config.yaml", "r") as f:
        config = yaml.full_load(f)

    # Build network
    network = build_network(config)
    # Load network parameters
    network.load_state_dict(torch.load(parameters))

    # Two-layer
    if isinstance(network, TwoLayerSNN):
        network.fc1.weight.clamp_(-256, 254)
        network.fc2.weight.clamp_(-256, 254)

    # Three-layer
    elif isinstance(network, ThreeLayerSNN):
        network.fc1.weight.clamp_(-256, 254)
        network.fc2.weight.clamp_(-256, 254)
        network.fc3.weight.clamp_(-256, 254)

    # Save
    torch.save(network.state_dict(), parameters.strip(".net") + "_clamped.net")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    weights_to_csv(args["folder"], args["parameters"])
