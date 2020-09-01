import argparse
import yaml
import os
import shutil

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

mpl.rcParams["lines.linewidth"] = 0.8

from evolutionary.network.snn import EncodingOnly


def plot_ss_enc(folder, parameters):
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder + f"/steadystate+{individual_id}"
    suffix = 0
    while os.path.exists(f"{save_folder}+{str(suffix)}/"):
        suffix += 1
    save_folder += f"+{str(suffix)}/"
    os.makedirs(save_folder)

    # Load config
    with open(folder + "/config.yaml", "r") as cf:
        config = yaml.full_load(cf)

    # Load network
    network = EncodingOnly(config)
    network.load_state_dict(torch.load(parameters))
    network.reset_state()

    # Input: D in [-10, 10], Ddot in [-20, 20]
    div_lim = [-10.0, 10.0]
    divdot_lim = [-20.0, 20.0]
    div = np.linspace(*div_lim, 101)
    divdot = np.linspace(*divdot_lim, 201)

    # Batch size is 1, so no parallel stuff
    # Show each input for 100 steps
    time = 100
    spike_response = np.zeros(
        (div.shape[0], divdot.shape[0], config["net"]["layer sizes"][0], time)
    )
    trace_response = np.zeros(
        (div.shape[0], divdot.shape[0], config["net"]["layer sizes"][0], time)
    )

    # Start loop
    for i in range(div.shape[0]):
        for j in range(divdot.shape[0]):
            network.reset_state()
            for k in range(time):
                obs = np.array([div[i], divdot[j]])
                obs = torch.from_numpy(obs).float()
                spike, trace = network.forward(obs.view(1, 1, -1))
                spike_response[i, j, :, k] = spike.detach().clone()
                trace_response[i, j, :, k] = trace.detach().clone()

    # Interpolate
    x = np.linspace(*div_lim, 401)
    y = np.linspace(*divdot_lim, 801)
    xi, yi = np.meshgrid(x, y)
    spike_response = spike_response[:, :, :, -50:].mean(-1)
    trace_response = trace_response[:, :, :, -50:].mean(-1)

    for i in range(config["net"]["layer sizes"][0]):
        interp = RectBivariateSpline(div, divdot, spike_response[..., i])
        zi = interp.ev(xi, yi)

        # Save raw response
        # divi, divdoti = np.meshgrid(div, divdot)
        # data = pd.DataFrame(
        #     {"x": divi.flatten(), "y": divdoti.flatten(), "z": spike_response[..., i].T.flatten()}
        # )
        # data.to_csv(save_folder + "ss_raw.csv", index=False, sep=",")

        # Visualize
        fig, ax = plt.subplots(1, 1)

        # Bounded
        im = ax.imshow(
            zi,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            extent=[*div_lim, *divdot_lim],
            aspect=0.5,
            origin="lower",
        )
        ax.set_title(f"steady state response {i}")
        ax.set_ylabel("divergence dot [1/s2]")
        ax.set_xlabel("divergence [1/s]")
        ax.set_xlim(div_lim)
        ax.set_ylim(divdot_lim)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("spike mean")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_ss_enc(args["folder"], args["parameters"])
