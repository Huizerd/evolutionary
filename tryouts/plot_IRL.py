import argparse
import glob

import pandas as pd
import matplotlib.pyplot as plt


def plot_IRL(folder):
    # Find files
    files = glob.glob(folder + "*.csv")

    # Create plot
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
    axs[0].set_ylabel("height [m]")
    axs[1].set_ylabel("velocity [m/s]")
    axs[2].set_ylabel("divergence [1/s]")
    axs[3].set_ylabel("divergence dot [1/s2]")
    axs[4].set_ylabel("thrust [m/s2]")
    axs[4].set_xlabel("time [s]")

    # Go over files
    i = 0
    for f in files:
        # Read data
        data = pd.read_csv(f, sep=",")

        # Find starts/stops of runs
        starts = data.index[data["record"].diff().eq(1)].tolist()
        stops = data.index[data["record"].diff().eq(-1)].tolist()

        # Go over runs in a file
        for start, stop in zip(starts, stops):
            # Select data and reset time from start
            plot_data = data.loc[
                start:stop,
                [
                    "time",
                    "pos_z",
                    "vel_z",
                    "div",
                    "div_gt",
                    "divdot",
                    "divdot_gt",
                    "acc_z",
                    "acc_lp",
                    "thrust",
                ],
            ]
            plot_data.reset_index(inplace=True)
            plot_data["time"] -= plot_data.loc[0, "time"]

            # Plot data
            axs[0].plot(plot_data["time"], -plot_data["pos_z"], label=f"run {i}")
            axs[1].plot(plot_data["time"], -plot_data["vel_z"], label=f"run {i}")
            axs[2].plot(plot_data["time"], plot_data["div"], label=f"run {i}")
            axs[2].plot(
                plot_data["time"], plot_data["div_gt"], "-.", label=f"run {i} GT"
            )
            axs[3].plot(plot_data["time"], plot_data["divdot"], label=f"run {i}")
            axs[3].plot(
                plot_data["time"], plot_data["divdot_gt"], "-.", label=f"run {i} GT"
            )
            axs[4].plot(
                plot_data["time"], -plot_data["acc_z"], ":", label=f"run {i} raw"
            )
            axs[4].plot(
                plot_data["time"], -plot_data["acc_lp"], "--", label=f"run {i} low-pass"
            )
            axs[4].plot(
                plot_data["time"], plot_data["thrust"], label=f"run {i} setpoint"
            )

            # Increment counter
            i += 1

    # Add grid and legend
    for ax in axs:
        ax.grid()
        ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_IRL(args["folder"])
