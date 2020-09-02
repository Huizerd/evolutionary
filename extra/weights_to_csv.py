import argparse
import os
import yaml

import torch
import pandas as pd

from pysnn.neuron import AdaptiveLIFNeuron

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
        # In -> hid, dimension (post, pre)
        inhid = pd.DataFrame(
            network.fc1.weight.data.numpy(),
            columns=[f"pre{i}" for i in range(network.fc1.weight.shape[1])],
            index=[f"post{i}" for i in range(network.fc1.weight.shape[0])],
        )
        inhid.to_csv(save_folder + "inhid.csv")
        # Hid -> out, dimension (post, pre)
        hidout = pd.DataFrame(
            network.fc2.weight.data.numpy(),
            columns=[f"pre{i}" for i in range(network.fc2.weight.shape[1])],
            index=[f"post{i}" for i in range(network.fc2.weight.shape[0])],
        )
        hidout.to_csv(save_folder + "hidout.csv")

    # Three-layer
    elif isinstance(network, ThreeLayerSNN):
        # In -> hid1, dimension (post, pre)
        inhid1 = pd.DataFrame(
            network.fc1.weight.data.numpy(),
            columns=[f"pre{i}" for i in range(network.fc1.weight.shape[1])],
            index=[f"post{i}" for i in range(network.fc1.weight.shape[0])],
        )
        inhid1.to_csv(save_folder + "inhid1.csv")
        # Hid1 -> hid2, dimension (post, pre)
        hid1hid2 = pd.DataFrame(
            network.fc2.weight.data.numpy(),
            columns=[f"pre{i}" for i in range(network.fc2.weight.shape[1])],
            index=[f"post{i}" for i in range(network.fc2.weight.shape[0])],
        )
        hid1hid2.to_csv(save_folder + "hid1hid2.csv")
        # Hid2 -> out, dimension (post, pre)
        hid2out = pd.DataFrame(
            network.fc3.weight.data.numpy(),
            columns=[f"pre{i}" for i in range(network.fc3.weight.shape[1])],
            index=[f"post{i}" for i in range(network.fc3.weight.shape[0])],
        )
        hid2out.to_csv(save_folder + "hid2out.csv")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    weights_to_csv(args["folder"], args["parameters"])
