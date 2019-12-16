import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def replot_sensitivity(folder, pareto=False):
    # Load data and masks
    folder = Path(folder)
    run = pd.read_csv(folder / "sensitivity.txt", sep="\t").to_numpy()
    if pareto:
        mask = np.load(folder / "mask_pareto.npy")
    else:
        mask = np.load(folder / "mask.npy")

    run = run[mask]

    # Plot results
    fig1, ax1 = plt.subplots(1, 1, dpi=200)
    ax1.set_title("Performance sensitivity")
    ax1.set_xlabel("time to land")
    ax1.set_ylabel("final velocity")
    ax1.set_xlim([0.0, 10.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.grid()

    # Scatter plot with error bars for 25th and 75th
    ax1.errorbar(
        run[:, 4],
        run[:, 6],
        xerr=np.abs((run[:, [0, 8]] - run[:, 4][..., None]).T),
        yerr=np.abs((run[:, [2, 10]] - run[:, 6][..., None]).T),
        linestyle="",
        marker="",
        color="k",
        elinewidth=0.5,
        capsize=1,
        capthick=0.5,
        zorder=10,
    )
    cb = ax1.scatter(
        run[:, 4],
        run[:, 6],
        marker=".",
        c=run[:, 7],
        cmap="coolwarm",
        s=np.abs(run[:, 11] - run[:, 3]),
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
    ax2.set_xlim([0.0, 10.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.grid()

    # Scatter plot with error bars for 25th and 75th
    for i in range(run.shape[0]):
        ax2.text(run[i, 4], run[i, 6], str(int(run[i, -1])), fontsize=5)
    fig2.colorbar(cb, ax=ax2)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--pareto", action="store_true")
    args = vars(parser.parse_args())

    # Call
    replot_sensitivity(args["folder"], pareto=args["pareto"])
