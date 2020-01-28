import argparse

import pandas as pd
import matplotlib.pyplot as plt


def process_thrust(folder):
    # For all 5 runs
    for i in range(5):
        run = pd.read_csv(folder + f"run{i}.csv", sep=",")
        fig, ax = plt.subplots(1, 1)
        lowpassed = run["tsp"].rolling(window=20, min_periods=1).mean()
        ax.plot(run["time"], run["tsp"], "b")
        ax.plot(run["time"], lowpassed, "r")

        run["tsp_lp"] = lowpassed
        run.to_csv(folder + f"run{i}.csv", index=False, sep=",")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    process_thrust(args["folder"])
