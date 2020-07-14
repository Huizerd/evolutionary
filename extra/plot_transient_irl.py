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

mpl.rcParams["lines.linewidth"] = 0.8

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env


def plot_transient_irl(folder, plot_only):
    folder = Path(folder)

    # Go over all IRL runs
    runs = sorted(Path(folder).rglob("run?.csv"))

    # Lists for the blue background dots (unsorted/unprocessed)
    all_x = []
    all_y = []
    # Figure for plotting
    fig, ax = plt.subplots(1, 1)

    for run in runs:
        # Extract index
        i = run.stem[-1]
        # Check for int
        try:
            int(i)
        except ValueError:
            print("Run indices are not clear!")

        # Load run
        run = pd.read_csv(run, sep=",")

        # Sort by increasing divergence
        sort_idx = np.argsort(run["div"])
        x = run["div"].values[sort_idx]
        y = run["thrust"].values[sort_idx]

        # Take moving average of sorted for the red lines
        ma_y = pd.Series(y).rolling(window=40, min_periods=1).mean().values

        # Plot line
        ax.plot(x, ma_y, "r", alpha=0.5)

        # Add to lists of blue dots
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

        # Write to dataframe
        if not plot_only:
            output = pd.DataFrame({"x": x, "y": ma_y})
            output.to_csv(folder / f"run{i}_transient.csv", index=False, sep=",")

    # Plot blue dots
    # Do these go in the background by default?
    ax.scatter(all_x, all_y, c="b", alpha=0.5)
    ax.grid()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-0.9, 0.9])
    fig.tight_layout()
    plt.show()

    # Write to dataframe as well
    if not plot_only:
        scatter = pd.DataFrame({"x": all_x, "y": all_y})
        scatter.to_csv(folder / f"raw_points_transient.csv", index=False, sep=",")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--plot_only", action="store_true")
    args = vars(parser.parse_args())

    # Call
    plot_transient_irl(args["folder"], args["plot_only"])
