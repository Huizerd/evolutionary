import argparse
import os
import yaml

import torch
import pandas as pd
import matplotlib.pyplot as plt

from evolutionary.utils.constructors import build_network


def plot_weights(folder, parameters):
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

    # Create figure
    fig, ax = plt.subplots(2, 2)

    # In -> hid
    # Square
    im1 = ax[0, 0].imshow(
        network.fc1.weight.data.numpy(), vmin=-256, vmax=254, cmap="viridis"
    )
    ax[0, 0].set_xlabel("pre id")
    ax[0, 0].set_ylabel("post id")
    ax[0, 0].set_title("in -> hid")
    cbar1 = fig.colorbar(im1, ax=ax[0, 0])
    cbar1.set_label("weight")
    # Histogram
    ax[0, 1].hist(
        network.fc1.weight.data.view(-1).numpy(),
        range=[-256, 254],
        bins=20,
        edgecolor="black",
    )
    ax[0, 1].set_xlabel("weight")
    ax[0, 1].set_ylabel("count")
    ax[0, 1].set_title("in -> hid")

    # Hid -> out
    # Square
    im2 = ax[1, 0].imshow(
        network.fc2.weight.data.numpy(), vmin=-256, vmax=254, cmap="viridis"
    )
    ax[1, 0].set_xlabel("pre id")
    ax[1, 0].set_ylabel("post id")
    ax[1, 0].set_title("hid -> out")
    cbar2 = fig.colorbar(im2, ax=ax[1, 0])
    cbar2.set_label("weight")
    # Histogram
    ax[1, 1].hist(
        network.fc2.weight.data.view(-1).numpy(),
        range=[-256, 254],
        bins=20,
        edgecolor="black",
    )
    ax[1, 1].set_xlabel("weight")
    ax[1, 1].set_ylabel("count")
    ax[1, 1].set_title("hid -> out")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_weights(args["folder"], args["parameters"])
