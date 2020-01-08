import argparse

import pandas as pd
import numpy as np
from scipy import stats


def get_r_values(folder):
    # For all 5 runs, comparison between hidden and output neuron spike trace
    r_values = pd.DataFrame(columns=["pair", "r0", "r1", "r2", "r3", "r4", "avg"])
    run = pd.read_csv(folder + f"run0.csv", sep=",")

    # Number of neurons
    neurons = 0
    for col in run.columns:
        if "_ma" in col:
            neurons += 1

    for id in range(neurons - 1):
        rrow = {"pair": f"n{id}-out"}
        for i in range(5):
            run = pd.read_csv(folder + f"run{i}.csv", sep=",")
            _, _, rv, _, _ = stats.linregress(
                run[f"n{id}_ma"], run[f"n{neurons - 1}_ma"]
            )
            rrow[f"r{i}"] = rv ** 2

        rrow["avg"] = np.mean([rrow[f"r{j}"] for j in range(5)])

        r_values = r_values.append(rrow, ignore_index=True)

    r_values.to_csv(folder + "r_values.csv", index=False, sep=",")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    get_r_values(args["folder"])
