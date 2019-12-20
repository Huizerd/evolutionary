import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_opt_perf(folder):
    files = Path(folder).rglob("optim_performance.txt")

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("convergence")
    ax[0].set_xlabel("generation")
    ax[0].grid()
    ax[1].set_title("hypervolume")
    ax[1].set_xlabel("generation")
    ax[1].grid()
    for i, f in enumerate(files):
        data = pd.read_csv(f, sep="\t")
        ax[0].plot(data["convergence"], label=str(f).split("/")[-2])
        ax[1].plot(data["hypervolume"], label=str(f).split("/")[-2])

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_opt_perf(args["folder"])
