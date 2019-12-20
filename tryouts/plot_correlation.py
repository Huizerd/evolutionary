import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def plot_correlation(folder, filter=False, pareto=False):
    # Load data and masks
    folder = Path(folder)
    performance = np.load(folder / "raw_performance.npy")
    if filter:
        if pareto:
            mask = np.load(folder / "mask_pareto.npy")
            performance = performance[mask, :, :]
        else:
            mask = np.load(folder / "mask.npy")
            performance = performance[mask, :, :]

    # Compute mean
    mean = np.mean(performance, 1)

    # Compute variance and IQR
    q25, q75 = np.percentile(performance, [25, 75], 1)
    iqr = q75 - q25
    var = np.var(performance, 1)

    # Compute Pearson correlation and Spearman correlation (allows non-linear)
    pearson_iqr = [
        pearsonr(iqr[:, i], iqr[:, -1]) for i in range(performance.shape[-1] - 1)
    ]
    pearson_var = [
        pearsonr(var[:, i], var[:, -1]) for i in range(performance.shape[-1] - 1)
    ]
    spearman_iqr = [
        spearmanr(iqr[:, i], iqr[:, -1]) for i in range(performance.shape[-1] - 1)
    ]
    spearman_var = [
        spearmanr(var[:, i], var[:, -1]) for i in range(performance.shape[-1] - 1)
    ]
    # pearson_iqr = [pearsonr(iqr[:, i], mean[:, -1]) for i in range(performance.shape[-1])]
    # pearson_var = [pearsonr(var[:, i], mean[:, -1]) for i in range(performance.shape[-1])]
    # spearman_iqr = [spearmanr(iqr[:, i], mean[:, -1]) for i in range(performance.shape[-1])]
    # spearman_var = [spearmanr(var[:, i], mean[:, -1]) for i in range(performance.shape[-1])]

    print(pearson_iqr)
    print(pearson_var)
    print(spearman_iqr)
    print(spearman_var)


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--pareto", action="store_true")
    args = vars(parser.parse_args())

    # Call
    plot_correlation(args["folder"], filter=args["filter"], pareto=args["pareto"])
