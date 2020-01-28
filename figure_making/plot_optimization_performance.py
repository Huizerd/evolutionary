import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def get_opt_perf(folder, ext):
    folder = Path(folder)
    files = folder.rglob(f"optim_performance*.{ext}")

    data = []
    for i, f in enumerate(files):
        # Divide by max hypervolume to get normalised
        if ext == "csv":
            run = pd.read_csv(f, sep=",")["hypervolume"].values / 1e7
        elif ext == "txt":
            run = pd.read_csv(f, sep="\t")["hypervolume"].values / 1e7
        data.append(run)
    mean = np.array(data).mean(0)
    std = np.array(data).std(0)
    stats = pd.DataFrame({"gen": range(mean.shape[0]), "mean": mean, "std": std})
    stats.to_csv(folder / "optim_hypervolume.csv", index=False, sep=",")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--ext", type=str, default="csv")
    args = vars(parser.parse_args())

    # Call
    get_opt_perf(args["folder"], args["ext"])
