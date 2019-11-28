import glob

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pysnn.network import SNNNetwork

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env, is_pareto_efficient


def vis_sensitivity(config, parameters, verbose=2):
    # Check objectives
    assert all(
        [
            obj1 == obj2
            for obj1, obj2 in zip(
                config["evo"]["objectives"][:3],
                ["time to land", "final height", "final velocity"],
            )
        ]
    ), "First 3 objectives should be time to land, final height and final velocity"
    # Expand to all parameter files
    # Also load fitnesses to compare consistency of our optimization
    # These should both be in the same order!
    with open(parameters + "fitnesses.txt", "r") as f:
        fitnesses = pd.read_csv(f, sep="\t")
        fitnesses["index"] = fitnesses.index
        fitnesses = fitnesses.to_numpy()
    parameters = sorted(glob.glob(parameters + "*.net"))

    # Build environment
    env = build_environment(config)

    # Build network
    network = build_network(config)

    # Performance over 100 runs
    performance = np.zeros((len(parameters), 100, 3))

    # Go over runs
    # We want all nets to be exposed to the same conditions in a single run
    for j in range(performance.shape[1]):
        # Randomize environment
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
            while not done:
                # Step environment
                obs = torch.from_numpy(obs)
                action = network.forward(obs.view(1, 1, -1))
                action = action.numpy()
                obs, _, done, _ = env.step(action)

            # Increment counters
            performance[i, j, :] = [
                env.t - config["env"]["settle"],
                env.state[0],
                abs(env.state[1]),
            ]

    # Process results: get median and 25th and 75th percentiles
    percentiles = np.percentile(performance, [25, 50, 75], 1)
    stds = np.std(performance, 1)
    print(
        f"Mean sigmas for time: {stds.mean(0)[0]:.3f}; height: {stds.mean(0)[1]:.3f}; velocity: {stds.mean(0)[2]:.3f}"
    )

    # Save results
    # Before filtering!
    if verbose:
        pd.DataFrame(
            stds, columns=["time to land", "final height", "final velocity"]
        ).to_csv(f"{config['log location']}sensitivity_stds.txt", index=False, sep="\t")
        pd.DataFrame(
            np.concatenate(
                [
                    fitnesses[:, :3],
                    percentiles[0, :, :],
                    percentiles[1, :, :],
                    percentiles[2, :, :],
                ],
                axis=1,
            ),
            columns=[
                "time to land",
                "final height",
                "final velocity",
                "25th_ttl",
                "25th_fh",
                "25th_fv",
                "50th_ttl",
                "50th_fh",
                "50th_fv",
                "75th_ttl",
                "75th_fh",
                "75th_fv",
            ],
        ).to_csv(f"{config['log location']}sensitivity.txt", index=False, sep="\t")

    # Filter results
    mask = (percentiles[1, :, 0] < 10.0) & (percentiles[1, :, 2] < 1.0)
    efficient = is_pareto_efficient(percentiles[1, :, :])
    fitnesses = fitnesses[mask & efficient, :]
    percentiles = percentiles[:, mask & efficient, :]

    # Plot results
    fig, ax = plt.subplots(1, 1, dpi=200)
    ax.set_title("Performance sensitivity")
    ax.set_xlabel(config["evo"]["objectives"][0])
    ax.set_ylabel(config["evo"]["objectives"][2])
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid()
    # Rectangles for 25th and 75th
    for i in range(percentiles.shape[1]):
        rect = patches.Rectangle(
            (percentiles[0, i, 0], percentiles[0, i, 2]),
            percentiles[2, i, 0] - percentiles[0, i, 0],
            percentiles[2, i, 2] - percentiles[0, i, 2],
            linewidth=0.5,
            edgecolor="k",
            facecolor="none",
        )
        ax.add_patch(rect)
        # And connect median to fitness
        ax.plot(
            [fitnesses[i, 0], percentiles[1, i, 0]],
            [fitnesses[i, 2], percentiles[1, i, 2]],
            "k:",
            linewidth=0.5,
        )
        # And annotate
        ax.text(
            percentiles[1, i, 0],
            percentiles[1, i, 2],
            str(int(fitnesses[i, -1])),
            va="top",
            fontsize=7,
        )

    # Medians
    ax.scatter(percentiles[1, :, 0], percentiles[1, :, 2], s=6)
    # Old fitnesses
    ax.scatter(fitnesses[:, 0], fitnesses[:, 2], s=6)
    fig.tight_layout()

    # Save figure
    if verbose:
        fig.savefig(f"{config['log location']}sensitivity.png")

    # Show figure
    if verbose > 1:
        plt.show()
