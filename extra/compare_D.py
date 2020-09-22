import argparse
import glob

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.5


def compare_D(sim_folder, real_folder):
    # Find files
    sim_files = glob.glob(sim_folder + "/run?.csv")
    real_files = glob.glob(real_folder + "/run?.csv")

    # Create plot
    fig, axs = plt.subplots(7, 2, sharex=True, figsize=(10, 10), dpi=200)
    axs[0, 0].set_title("sim")
    axs[0, 1].set_title("real")
    axs[0, 0].set_ylabel("height [m]")
    axs[1, 0].set_ylabel("velocity [m/s]")
    axs[2, 0].set_ylabel("acceleration [m/s2]")
    axs[3, 0].set_ylabel("thrust [g]")
    axs[4, 0].set_ylabel("divergence [1/s]")
    axs[5, 0].set_ylabel("divergence dot [1/s2]")
    axs[6, 0].set_ylabel("spikes [?]")
    axs[6, 0].set_xlabel("time [s]")
    axs[6, 1].set_xlabel("time [s]")

    # cm = plt.get_cmap("gist_rainbow")
    # for ax in axs.flatten():
    #     ax.set_prop_cycle(color=[cm(i / 15) for i in range(15)])

    # Simulation
    # Go over files
    for i, f in enumerate(sim_files):
        # Read data
        data = pd.read_csv(f, sep=",")

        # Plot data
        # Height
        axs[0, 0].plot(data["time"], data["pos_z"], label=f"run {i}")
        # Velocity
        axs[1, 0].plot(data["time"], data["vel_z"], label=f"run {i}")
        # Acceleration
        axs[2, 0].plot(data["time"], data["thrust"], label=f"run {i}")
        # Thrust
        axs[3, 0].plot(data["time"], data["tsp"], label=f"run {i}")
        # Divergence
        axs[4, 0].plot(data["time"], data["div"], label=f"run {i}")
        # Divergence dot
        axs[5, 0].plot(data["time"], data["divdot"], label=f"run {i}")
        # Spikes
        axs[6, 0].plot(
            data["time"], data["spike_count"] / data["time"], label=f"run {i}",
        )

    # Real world
    # Go over files
    for i, f in enumerate(real_files):
        # Read data
        data = pd.read_csv(f, sep=",")

        # Plot data
        # Height
        axs[0, 1].plot(data["time"], -data["pos_z"], label=f"run {i}")
        # Velocity
        axs[1, 1].plot(data["time"], -data["vel_z"], label=f"run {i}")
        # Acceleration
        axs[2, 1].plot(
            data["time"],
            -data["acc_lp"].rolling(window=5, min_periods=1).mean(),
            label=f"run {i}",
        )
        # axs[2, 1].plot(data["time"][1:].values, -pd.Series((data["vel_z"][1:].values - data["vel_z"][:-1].values) / (data["time"][1:].values - data["time"][:-1].values)).rolling(window=5, min_periods=1).mean(), label=f"run {i}")
        # Thrust
        axs[3, 1].plot(data["time"], data["thrust"], label=f"run {i}")
        # Divergence
        axs[4, 1].plot(data["time"], data["div"], label=f"run {i}")
        # Divergence dot
        axs[5, 1].plot(data["time"], data["divdot"], label=f"run {i}")
        # Spikes
        axs[6, 1].plot(
            data["time"], data["spike_count"] / data["time"], label=f"run {i}",
        )

    # Add grid
    for ax in axs.flatten():
        ax.grid()
    # Set limits
    for ax, ylim in zip(
        axs, [[0, 4], [-1.0, 0.5], [-3, 3], [-0.1, 0.1], [-1, 2], [-40, 40], [0, 1000]]
    ):
        for a in ax:
            a.set_ylim(ylim)
    fig.tight_layout()

    # Compare D for first 5 landings
    fig2, axs2 = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(10, 10), dpi=200)
    for ax in axs2:
        ax.set_ylabel("D true [1/s]")
    axs2[4].set_xlabel("D est [1/s]")

    for i, ax in enumerate(axs2):
        data_sim = pd.read_csv(sim_files[i], sep=",")
        try:
            data_real = pd.read_csv(real_files[i], sep=",")
        except IndexError:
            continue
        ax.scatter(data_sim["div"], data_sim["div_gt"], s=6, label="sim")
        ax.scatter(data_real["div"], data_real["div_gt"], s=6, label="real")
        ax.grid()

    axs2[0].legend()
    fig2.tight_layout()

    # Compare acceleration for first 5 landings
    fig3, axs3 = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(10, 10), dpi=200)
    for ax in axs3:
        ax.set_ylabel("a desired [m/s2]")
    axs3[4].set_xlabel("a actual [m/s2]")

    for i, ax in enumerate(axs3):
        data_sim = pd.read_csv(sim_files[i], sep=",")
        try:
            data_real = pd.read_csv(real_files[i], sep=",")
        except IndexError:
            continue
        ax.scatter(
            (data_sim["vel_z"][1:].values - data_sim["vel_z"][:-1].values)
            / (data_sim["time"][1:].values - data_sim["time"][:-1].values),
            data_sim["tsp"][1:].values * 9.81,
            s=6,
            label="sim",
        )
        # ax.scatter(data_sim["thrust"].values, data_sim["tsp"].values * 9.81, s=6)
        ax.scatter(
            -(data_real["vel_z"][1:].values - data_real["vel_z"][:-1].values)
            / (data_real["time"][1:].values - data_real["time"][:-1].values),
            data_real["thrust"][1:].values * 9.81,
            s=6,
            label="real: dvel/dt, thrust",
        )
        ax.scatter(
            -data_real["acc_lp"],
            data_real["thrust"] * 9.81,
            s=6,
            label="real: acc_lp, thrust",
        )
        ax.grid()

    axs3[0].legend()
    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--simf", type=str, required=True)
    parser.add_argument("--realf", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    compare_D(args["simf"], args["realf"])
