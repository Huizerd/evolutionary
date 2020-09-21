import argparse
import glob

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8


def plot_irl(folder):
    # Find files
    files = glob.glob(folder + "20*.csv")

    # Create plot
    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(10, 10), dpi=200)
    axs[0].set_ylabel("height [m]")
    axs[1].set_ylabel("velocity [m/s]")
    axs[2].set_ylabel("thrust [g]")
    axs[3].set_ylabel("thrust lp [g]")
    axs[4].set_ylabel("divergence [1/s]")
    axs[5].set_ylabel("divergence dot [1/s2]")
    axs[6].set_ylabel("spikes [?]")
    axs[6].set_xlabel("time [s]")

    cm = plt.get_cmap("gist_rainbow")
    for ax in axs:
        ax.set_prop_cycle(color=[cm(i / 15) for i in range(15)])

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
            plot_data["timestamp"] = plot_data["time"]
            plot_data["time"] -= plot_data.loc[0, "time"]
            plot_data["spike_step"] = plot_data["spike_count"].diff()
            plot_data["spike_step"][0] = 0
            plot_data["spike_step"] = plot_data["spike_step"].astype(int)
            plot_data["thrust"] = plot_data["thrust"] / 9.81
            plot_data["thrust_lp"] = (
                plot_data["thrust"].rolling(window=20, min_periods=1).mean()
            )
            # Earlier "landing"
            stop_alt = (-plot_data["pos_z"] < 0.1).idxmax()
            plot_data = plot_data.iloc[:stop_alt, :]

            # Plot data
            # Height
            axs[0].plot(plot_data["time"], -plot_data["pos_z"], label=f"run {i}")
            # Velocity
            axs[1].plot(plot_data["time"], -plot_data["vel_z"], label=f"run {i}")
            # Thrust
            axs[2].plot(plot_data["time"], plot_data["thrust"], label=f"run {i}")
            # Thrust low-pass
            axs[3].plot(plot_data["time"], plot_data["thrust_lp"], label=f"run {i}")
            # Divergence
            axs[4].plot(plot_data["time"], plot_data["div"], label=f"run {i}")
            # axs[3].plot(
            #     plot_data["time"], plot_data["div_gt"], "C1", label=f"run {i} GT"
            # )
            # Divergence dot
            axs[5].plot(plot_data["time"], plot_data["divdot"], label=f"run {i}")
            # axs[4].plot(
            #     plot_data["time"], plot_data["divdot_gt"], "C1", label=f"run {i} GT"
            # )
            # Spikes
            axs[6].plot(
                plot_data["time"],
                plot_data["spike_count"] / plot_data["time"],
                label=f"run {i}",
            )

            # Save data
            plot_data.to_csv(str(folder) + f"run{i}.csv", index=False, sep=",")

            # Increment counter
            i += 1

    # Add grid and legend
    for ax in axs:
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.grid()
    axs[0].legend()
    # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_irl(args["folder"])
