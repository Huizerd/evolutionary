from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysnn.network import SNNNetwork

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env, is_pareto_efficient


def vis_sensitivity_complete(config, parameters, verbose=2):
    # Expand to all parameter files
    # In order to combine multiple evolution runs: put them as subdirectories in one
    # big folder and use that as parameter argument
    parameters = sorted(Path(parameters).rglob("*.net"))

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
            # Only test from 5m
            obs = env.reset(h0=(config["env"]["h0"][0] + config["env"]["h0"][-1]) / 2)
            env.seed(env.seeds)
            if isinstance(network, SNNNetwork):
                network.reset_state()

            # Start run
            done = False
            spikes = 0
            while not done:
                # Step environment
                obs = torch.from_numpy(obs)
                action = network.forward(obs.view(1, 1, -1))
                action = action.numpy()
                obs, _, done, _ = env.step(action)
                if isinstance(network, SNNNetwork):
                    spikes += (
                        network.neuron1.spikes.sum().item()
                        + network.neuron2.spikes.sum().item()
                    )

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
    print(
        f"Mean sigmas for time: {stds.mean(0)[0]:.3f}; height: {stds.mean(0)[1]:.3f}; velocity: {stds.mean(0)[2]:.3f}; spikes: {stds.mean(0)[3]:.3f}"
    )

    # Save results
    # Before filtering!
    if verbose:
        pd.DataFrame(
            stds, columns=["time to land", "final height", "final velocity", "spikes"]
        ).to_csv(f"{config['log location']}sensitivity_stds.txt", index=False, sep="\t")
        pd.DataFrame(
            np.concatenate(
                [percentiles[0, :, :], percentiles[1, :, :], percentiles[2, :, :]],
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
            ],
        ).to_csv(f"{config['log location']}sensitivity.txt", index=False, sep="\t")
        # Also save raw performance as npz
        np.save(f"{config['log location']}raw_performance", performance)

    # Filter results
    mask = (percentiles[1, :, 0] < 10.0) & (percentiles[1, :, 2] < 1.0)
    efficient = is_pareto_efficient(percentiles[1, :, :])
    percentiles = percentiles[:, mask & efficient, :]

    # Plot results
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.set_title("Performance sensitivity")
    ax.set_xlabel(config["evo"]["objectives"][0])
    ax.set_ylabel(config["evo"]["objectives"][2])
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid()

    # Scatter plot with error bars for 25th and 75th
    ax.errorbar(
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
    cb = ax.scatter(
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

    fig.colorbar(cb, ax=ax)
    fig.tight_layout()

    # Save figure
    if verbose:
        fig.savefig(f"{config['log location']}sensitivity.png")

    # Show figure
    if verbose > 1:
        plt.show()
