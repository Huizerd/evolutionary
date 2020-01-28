import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from evolutionary.utils.utils import is_pareto_efficient


def compare_analyses(folder1, folder2, pareto=False):
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    run1 = np.load(folder1 / "raw_performance.npy")
    run2 = np.load(folder2 / "raw_performance.npy")

    # Compute percentiles
    perc1 = np.percentile(run1, [25, 50, 75, 90], 1)
    perc2 = np.percentile(run2, [25, 50, 75, 90], 1)
    # Compute stds
    stds1 = np.std(run1, 1)
    stds2 = np.std(run2, 1)
    # Filter
    mask1 = (perc1[1, :, 0] < 10.0) & (perc1[1, :, 2] < 1.0) & (stds1[:, 1] == 0.0)
    mask2 = (perc2[1, :, 0] < 10.0) & (perc2[1, :, 2] < 1.0) & (stds2[:, 1] == 0.0)
    efficient1 = is_pareto_efficient(perc1[1, :, :])
    efficient2 = is_pareto_efficient(perc2[1, :, :])
    mask1_pareto = mask1 & efficient1
    mask2_pareto = mask2 & efficient2
    if pareto:
        perc1 = perc1[:, mask1_pareto, :]
        perc2 = perc2[:, mask2_pareto, :]
    else:
        perc1 = perc1[:, mask1, :]
        perc2 = perc2[:, mask2, :]
    # Normalize with hist below

    # Option to save masks here
    np.save(folder1 / "mask", mask1)
    np.save(folder2 / "mask", mask2)
    np.save(folder1 / "mask_pareto", mask1_pareto)
    np.save(folder2 / "mask_pareto", mask2_pareto)

    # Plot figure of medians
    objs = ["ttl", "fh", "fv", "s"]
    min_obj = [
        min([perc[1, :, i].min() for perc in [perc1, perc2]]) for i in range(len(objs))
    ]
    max_obj = [
        max([perc[1, :, i].max() for perc in [perc1, perc2]]) for i in range(len(objs))
    ]
    bins = [
        np.linspace(mn, mx, 15) if mn < mx else None for mn, mx in zip(min_obj, max_obj)
    ]
    fig, axs = plt.subplots(1, len(objs), figsize=(10, 5))
    for i, obj, ax, bn in zip(range(len(objs)), objs, axs, bins):
        ax.set_title(obj + " 50")
        ax.grid()
        ax.hist(
            perc1[1, :, i],
            bn,
            density=True,
            edgecolor="k",
            alpha=0.5,
            label="1",
            zorder=10,
        )
        ax.hist(
            perc2[1, :, i],
            bn,
            density=True,
            edgecolor="k",
            alpha=0.5,
            label="2",
            zorder=100,
        )
    axs[0].legend()
    fig.tight_layout()

    # Plot figure of 75th percentiles
    min_obj = [
        min([perc[2, :, i].min() for perc in [perc1, perc2]]) for i in range(len(objs))
    ]
    max_obj = [
        max([perc[2, :, i].max() for perc in [perc1, perc2]]) for i in range(len(objs))
    ]
    bins = [
        np.linspace(mn, mx, 15) if mn < mx else None for mn, mx in zip(min_obj, max_obj)
    ]
    fig, axs = plt.subplots(1, len(objs), figsize=(10, 5))
    for i, obj, ax, bn in zip(range(len(objs)), objs, axs, bins):
        ax.set_title(obj + " 75")
        ax.grid()
        ax.hist(
            perc1[2, :, i],
            bn,
            density=True,
            edgecolor="k",
            alpha=0.5,
            label="1",
            zorder=10,
        )
        ax.hist(
            perc2[2, :, i],
            bn,
            density=True,
            edgecolor="k",
            alpha=0.5,
            label="2",
            zorder=100,
        )
    axs[0].legend()
    fig.tight_layout()

    # Plot figure of 90th percentiles
    min_obj = [
        min([perc[3, :, i].min() for perc in [perc1, perc2]]) for i in range(len(objs))
    ]
    max_obj = [
        max([perc[3, :, i].max() for perc in [perc1, perc2]]) for i in range(len(objs))
    ]
    bins = [
        np.linspace(mn, mx, 15) if mn < mx else None for mn, mx in zip(min_obj, max_obj)
    ]
    fig, axs = plt.subplots(1, len(objs), figsize=(10, 5))
    for i, obj, ax, bn in zip(range(len(objs)), objs, axs, bins):
        ax.set_title(obj + " 90")
        ax.grid()
        ax.hist(
            perc1[3, :, i],
            bn,
            density=True,
            edgecolor="k",
            alpha=0.5,
            label="1",
            zorder=10,
        )
        ax.hist(
            perc2[3, :, i],
            bn,
            density=True,
            edgecolor="k",
            alpha=0.5,
            label="2",
            zorder=100,
        )
    axs[0].legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    parser.add_argument("--pareto", action="store_true")
    args = vars(parser.parse_args())

    # Call
    compare_analyses(args["folder1"], args["folder2"], pareto=args["pareto"])
