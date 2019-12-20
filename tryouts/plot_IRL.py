import argparse
import glob

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8


def plot_IRL(folder):
    # Find files
    files = glob.glob(folder + "2019*.csv")

    # Create plot
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
    axs[0].set_ylabel("height [m]")
    axs[1].set_ylabel("velocity [m/s]")
    axs[2].set_ylabel("thrust [m/s2]")
    axs[3].set_ylabel("divergence [1/s]")
    axs[4].set_ylabel("divergence dot [1/s2]")
    axs[5].set_ylabel("spikes [?]")
    axs[5].set_xlabel("time [s]")

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
                    "thrust",
                    "spike_count",
                ],
            ]
            plot_data.reset_index(drop=True, inplace=True)
            plot_data["time"] -= plot_data.loc[0, "time"]
            plot_data["spike_step"] = plot_data["spike_count"].diff()
            plot_data["spike_step"][0] = 0
            plot_data["spike_step"] = plot_data["spike_step"].astype(int)
            # Earlier "landing"
            stop_alt = (-plot_data["pos_z"] < 0.1).idxmax()
            plot_data = plot_data.iloc[:stop_alt, :]

            # Plot data
            # Height
            axs[0].plot(plot_data["time"], -plot_data["pos_z"], "C0", label=f"run {i}")
            # Velocity
            axs[1].plot(plot_data["time"], -plot_data["vel_z"], "C0", label=f"run {i}")
            # Thrust
            axs[2].plot(plot_data["time"], plot_data["thrust"], "C0", label=f"run {i}")
            # Divergence
            axs[3].plot(plot_data["time"], plot_data["div"], "C0", label=f"run {i}")
            axs[3].plot(
                plot_data["time"], plot_data["div_gt"], "C1", label=f"run {i} GT"
            )
            # Divergence dot
            axs[4].plot(plot_data["time"], plot_data["divdot"], "C0", label=f"run {i}")
            axs[4].plot(
                plot_data["time"], plot_data["divdot_gt"], "C1", label=f"run {i} GT"
            )
            # Spikes
            axs[5].plot(
                plot_data["time"],
                plot_data["spike_count"] / plot_data["time"],
                "C0",
                label=f"run {i}",
            )

            # Save data
            plot_data.to_csv(str(folder) + f"run{i}.csv", index=False, sep=",")

            # Increment counter
            i += 1

    # Add grid and legend
    for ax in axs:
        ax.grid()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_IRL(args["folder"])
