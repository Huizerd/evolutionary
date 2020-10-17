import argparse
import os
import yaml
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env, is_pareto_efficient


def plot_sensitivity(folder, parameters, show=False):
    save_folder = folder + "/analysis"
    suffix = 0
    while os.path.exists(f"{save_folder}+{str(suffix)}/"):
        suffix += 1
    save_folder += f"+{str(suffix)}/"
    os.makedirs(save_folder)

    # Expand to all parameter files
    # In order to combine multiple evolution runs: put them as subdirectories in one
    # big folder and use that as parameter argument
    assert os.path.isdir(parameters), "Provide a folder of parameters for analysis!"
    parameters = sorted(Path(parameters).rglob("*.net"))
    ids = np.arange(0, len(parameters))
    # Save parameters with indices as DataFrame for later identification of good controllers
    pd.DataFrame(
        [(i, p) for i, p in zip(ids, parameters)], columns=["id", "location"]
    ).to_csv(save_folder + "ids.csv", index=False, sep=",")

    # Load config
    with open(folder + "/config.yaml", "r") as f:
        config = yaml.full_load(f)

    # Build environment
    env = build_environment(config)

    # Build network
    network = build_network(config)

    # Performance over 250 runs
    # Record time to land, final height, final velocity and spikes per second
    performance = np.zeros((len(parameters), 250, 4))

    # Go over runs
    for j in range(performance.shape[1]):
        # Randomize environment here, because we want all nets to be exposed to the same
        # conditions in a single run
        env = randomize_env(env, config)

        # Go over all individuals
        for i, param in enumerate(parameters):
            # Load network
            network.load_state_dict(torch.load(param))

            # Reset env and net (may be superfluous)
            # Also reseed env to make noise equal across runs!
            # Only test from 3.5 m
            obs = env.reset(h0=(config["env"]["h0"][-1] + config["env"]["h0"][0]) / 2)
            env.seed(env.seeds)
            network.reset_state()

            # Start run
            done = False
            spikes = 0
            while not done:
                # Step environment
                obs = torch.from_numpy(obs)
                action = network.forward(obs.view(1, 1, -1))
                action = action.numpy()
                spikes += (
                    network.neuron1.spikes.sum().item()
                    + network.neuron2.spikes.sum().item()
                    if network.neuron1 is not None
                    else network.neuron2.spikes.sum().item()
                )

                obs, _, done, _ = env.step(action)

            # Increment counters
            performance[i, j, :] = [
                env.t - config["env"]["settle"],
                env.state[0],
                abs(env.state[1]),
                spikes / (env.t - config["env"]["settle"]),
            ]

    # Process results: get median and 25th and 75th percentiles
    percentiles = np.percentile(performance, [25, 50, 75], 1)
    stds = np.std(performance, 1)

    # Save results
    # Before filtering!
    pd.DataFrame(
        stds, columns=["time to land", "final height", "final velocity", "spikes"]
    ).to_csv(save_folder + "sensitivity_stds.csv", index=False, sep=",")
    pd.DataFrame(
        np.concatenate(
            [
                percentiles[0, :, :],
                percentiles[1, :, :],
                percentiles[2, :, :],
                ids[:, None],
            ],
            axis=1,
        ),
        columns=[
            "25th_ttl",
            "25th_fh",
            "25th_fv",
            "25th_s",
            "50th_ttl",
            "50th_fh",
            "50th_fv",
            "50th_s",
            "75th_ttl",
            "75th_fh",
            "75th_fv",
            "75th_s",
            "id",
        ],
    ).to_csv(save_folder + "sensitivity.csv", index=False, sep=",")
    # Also save raw performance as npz
    np.save(save_folder + "raw_performance", performance)

    # Filter results
    mask = (percentiles[1, :, 0] < 30.0) & (percentiles[1, :, 2] < 1.0)
    efficient = is_pareto_efficient(percentiles[1, :, :])
    mask_pareto = mask & efficient
    percentiles = percentiles[:, mask, :]
    ids = ids[mask]

    # Also save filters/masks as npy for later use
    np.save(save_folder + "mask", mask)
    np.save(save_folder + "mask_pareto", mask_pareto)

    # Plot results
    fig1, ax1 = plt.subplots(1, 1, dpi=200)
    ax1.set_title("Performance sensitivity")
    ax1.set_xlabel("time to land")
    ax1.set_ylabel("final velocity")
    ax1.set_xlim([5.0, 12.0])
    ax1.set_ylim([0.0, 0.2])
    ax1.grid()

    # Scatter plot with error bars for 25th and 75th
    ax1.errorbar(
        percentiles[1, :, 0],
        percentiles[1, :, 2],
        xerr=np.abs(percentiles[[0, 2], :, 0] - percentiles[1, :, 0]),
        yerr=np.abs(percentiles[[0, 2], :, 2] - percentiles[1, :, 2]),
        linestyle="",
        marker="",
        color="k",
        elinewidth=0.5,
        capsize=1,
        capthick=0.5,
        zorder=10,
    )
    cb = ax1.scatter(
        percentiles[1, :, 0],
        percentiles[1, :, 2],
        marker=".",
        c=percentiles[1, :, 3],
        cmap="coolwarm",
        s=np.abs(percentiles[2, :, 3] - percentiles[0, :, 3]),
        linewidths=0.5,
        edgecolors="k",
        vmin=None,
        vmax=None,
        zorder=100,
    )

    fig1.colorbar(cb, ax=ax1)
    fig1.tight_layout()

    # Also plot figure with IDs
    fig2, ax2 = plt.subplots(1, 1, dpi=200)
    ax2.set_title("Performance sensitivity")
    ax2.set_xlabel("time to land")
    ax2.set_ylabel("final velocity")
    ax2.set_xlim([5.0, 12.0])
    ax2.set_ylim([0.0, 0.2])
    ax2.grid()

    # Scatter plot with error bars for 25th and 75th
    for i in range(percentiles.shape[1]):
        ax2.text(percentiles[1, i, 0], percentiles[1, i, 2], str(ids[i]), fontsize=5)
    fig2.colorbar(cb, ax=ax2)
    fig2.tight_layout()

    # Save figure
    fig1.savefig(save_folder + "sensitivity.png")
    fig2.savefig(save_folder + "ids.png")

    # Show figure
    if show:
        plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    parser.add_argument("--show", action="store_true")
    args = vars(parser.parse_args())

    # Call
    plot_sensitivity(args["folder"], args["parameters"], args["show"])
