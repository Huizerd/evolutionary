import argparse
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation


# Initialization function: plot the background of each frame
def init(line1, line2):
    line1.set_data([], [])
    line2.set_data([], [])
    return (line1, line2)


# Animation function, this is called sequentially
def animate(data, line1, line2, i):
    x = data["time"]
    y1 = data["div"] - 1
    y2 = data["thrust"]
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    return (line1, line2)


# Main function
def animate_run(inp, out):
    data = pd.read_csv(inp, sep=",")

    # First set up the figure, the axis, and the plot element we want to animate
    fig, axs = plt.subplots(
        2, 1, figsize=(4, 6), sharex=True, dpi=400, subplot_kw={"xlim": (0, 10)}
    )
    # fig = plt.figure(figsize=(5, 3), dpi=400)
    # fig.patch.set_facecolor("#abd7e6")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.0)
    # fig.patch.set_alpha(0.0)
    axs[0].set_ylim(-2, 4)
    axs[1].set_ylim(-0.5, 0.2)
    # ax = plt.axes(xlim=(0, 2.5), ylim=(-0.9, 0.6))
    # axs[0].grid()
    # axs[1].grid()
    # ax.set_xticklabels([])
    axs[0].set_ylabel("Divergence error [s$^{-1}$]")
    axs[1].set_ylabel("Thrust setpoint [g]")
    axs[1].set_xlabel("Time [s]")
    # ax.set_xlabel("time [s]")
    for ax in axs:
        ax.grid()
        ax.set_facecolor("white")
        ax.set_alpha(0.0)
        ax.set_facecolor("white")
        ax.set_alpha(0.0)
        ax.spines["bottom"].set_color("black")
        ax.spines["top"].set_color("black")
        ax.spines["right"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        ax.yaxis.label.set_color("black")
        ax.xaxis.label.set_color("black")

    (line1,) = axs[0].plot(
        [],
        [],
        "r",
        # marker="^",
        # markersize=15,
        # markeredgecolor="white",
        # markerfacecolor="red",
        # linestyle=None,
        lw=1,
    )
    (line2,) = axs[1].plot(
        [],
        [],
        "r",
        # marker="^",
        # markersize=15,
        # markeredgecolor="white",
        # markerfacecolor="red",
        # linestyle=None,
        lw=1,
    )

    # Create partial functions
    animate_p = partial(animate, data, line1, line2)
    init_p = partial(init, line1, line2)

    # Call the animator; blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate_p, init_func=init_p, frames=len(data), interval=0, blit=True
    )

    fig.tight_layout()
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    print(f"real fps: {1 / (data['time'][len(data) - 1] / (len(data) - 1))}")
    anim.save(
        out,
        fps=1 / (data["time"][len(data) - 1] / (len(data) - 1)),
        codec="png",
        savefig_kwargs={"facecolor": "none", "transparent": True},
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = vars(parser.parse_args())

    animate_run(args["inp"], args["out"])
