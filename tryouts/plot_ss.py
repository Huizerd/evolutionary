import argparse
import yaml
import os
import shutil
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

mpl.rcParams["lines.linewidth"] = 0.8

from pysnn.network import SNNNetwork

from evolutionary.utils.constructors import build_network


def plot_ss(folder, parameters, runs):
    folder = Path(folder)
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder / ("steadystate+" + individual_id)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    # Get run filenames
    if runs is not None:
        runs = sorted(Path(runs).rglob("run*.csv"))

    # Load config
    with open(folder / "config.yaml", "r") as cf:
        config = yaml.full_load(cf)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))
    if isinstance(network, SNNNetwork):
        network.reset_state()

    # Input: D in [-10, 10], Ddot in [-20, 20]
    div_lim = [-10.0, 10.0]
    divdot_lim = [-20.0, 20.0]
    div = np.linspace(*div_lim, 101)
    divdot = np.linspace(*divdot_lim, 201)

    # Batch size is 1, so no parallel stuff
    # Show each input for 100 steps
    time = 100
    response = np.zeros((div.shape[0], divdot.shape[0], time))

    # Start loop
    for i in range(div.shape[0]):
        for j in range(divdot.shape[0]):
            if isinstance(network, SNNNetwork):
                network.reset_state()
            for k in range(time):
                obs = np.array([div[i], divdot[j]])
                obs = torch.from_numpy(obs).float()
                action = network.forward(obs.view(1, 1, -1))
                response[i, j, k] = action.item()

    # Interpolate
    x = np.linspace(*div_lim, 401)
    y = np.linspace(*divdot_lim, 801)
    xi, yi = np.meshgrid(x, y)
    response = response[:, :, -50:].mean(-1)
    interp = RectBivariateSpline(div, divdot, response)
    zi = interp.ev(xi, yi)

    # Save raw response
    divi, divdoti = np.meshgrid(div, divdot)
    data = pd.DataFrame(
        {"x": divi.flatten(), "y": divdoti.flatten(), "z": response.T.flatten()}
    )
    data.to_csv(str(save_folder) + f"/ss_raw.csv", index=False, sep=",")

    # Visualize
    fig, ax = plt.subplots(1, 1)

    # Bounded
    im = ax.imshow(
        zi,
        vmin=config["env"]["g bounds"][0],
        vmax=config["env"]["g bounds"][1],
        cmap="viridis",
        extent=[*div_lim, *divdot_lim],
        aspect=0.5,
        origin="lower",
    )
    if runs is not None:
        for run in runs[:1]:
            run = pd.read_csv(run, sep=",")
            ax.plot(run["div_gt"], run["divdot_gt"], "r")
    ax.set_title("steady state response (bounded)")
    ax.set_ylabel("divergence dot [1/s2]")
    ax.set_xlabel("divergence [1/s]")
    ax.set_xlim(div_lim)
    ax.set_ylim(divdot_lim)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("thrust command [g]")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    parser.add_argument("--runs", type=str, default=None)
    args = vars(parser.parse_args())

    # Call
    plot_ss(args["folder"], args["parameters"], args["runs"])
