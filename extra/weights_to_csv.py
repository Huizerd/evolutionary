import argparse
import os
import yaml

import torch
import pandas as pd

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


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    weights_to_csv(args["folder"], args["parameters"])
