import argparse
import glob

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8


def plot_accel(file):
    # Read data
    data = pd.read_csv(file, sep=",")

    # Plot
    plt.plot(data["time"], -data["acc_z"])
    plt.plot(data["time"], -data["acc_lp"].rolling(window=5, min_periods=1).mean())
    plt.plot(data["time"], -data["acc_sp"])
    plt.plot(data["time"], data["thrust_lp"])
    plt.ylim([-10, 10])
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_accel(args["file"])
