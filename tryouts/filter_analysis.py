import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evolutionary.utils.utils import is_pareto_efficient


def filter_analysis(folder):
    folder = Path(folder)
    run = np.load(folder / "raw_performance.npy")
    ids = pd.read_csv(folder / "ids.txt", sep="\t")["id"].to_numpy()

    # Compute percentiles
    perc = np.percentile(run, [25, 50, 75], 1)
    # Compute stds
    stds = np.std(run, 1)
    # Filter
    mask = (perc[1, :, 0] < 10.0) & (perc[1, :, 2] < 1.0) & (stds[:, 1] == 0.0)
    efficient = is_pareto_efficient(perc[1, :, :])
    mask_pareto = mask & efficient
    perc_filtered = perc[:, mask, :]
    perc_pareto = perc[:, mask_pareto, :]
    ids_filtered = ids[mask]
    ids_pareto = ids[mask_pareto]

    # Option to save masks here
    np.save(folder / "mask", mask)
    np.save(folder / "mask_pareto", mask_pareto)

    # Save filtered percentiles
    pd.DataFrame(
        np.concatenate(
            [
                perc_filtered[0, :, :],
                perc_filtered[1, :, :],
                perc_filtered[2, :, :],
                ids_filtered[:, None],
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
    ).to_csv(folder / "sensitivity_filtered.txt", index=False, sep="\t")
    pd.DataFrame(
        np.concatenate(
            [
                perc_pareto[0, :, :],
                perc_pareto[1, :, :],
                perc_pareto[2, :, :],
                ids_pareto[:, None],
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
    ).to_csv(folder / "sensitivity_pareto.txt", index=False, sep="\t")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    filter_analysis(args["folder"])
