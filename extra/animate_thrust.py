import argparse
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation

# Initialization function: plot the background of each frame
def init(line1, line2, line3):
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return (line1, line2, line3)


# Animation function, this is called sequentially
def animate(data, line1, line2, line3, i):
    x = data["time"][:i]
    y1 = data["div"][:i]
    y2 = data["divdot"][:i]
    y3 = data["thrust"][:i]
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    line3.set_data(x, y3)
    return (line1, line2, line3)


# Main function
def animate_run(inp, out, fps):
    data = pd.read_csv(inp, sep=",")

    # First set up the figure, the axis, and the plot element we want to animate
    fig, axs = plt.subplots(
        3, 1, figsize=(4, 6), sharex=True, dpi=400, subplot_kw={"xlim": (0, 2.5)}
    )
    # fig = plt.figure(figsize=(5, 3), dpi=400)
    # fig.patch.set_facecolor("#abd7e6")
    fig.patch.set_facecolor("black")
    # fig.patch.set_alpha(0.0)
    axs[0].set_ylim(-10, 10)
    axs[1].set_ylim(-100, 100)
    axs[2].set_ylim(-0.9, 0.6)
    # ax = plt.axes(xlim=(0, 2.5), ylim=(-0.9, 0.6))
    # axs[0].grid()
    # axs[1].grid()
    # ax.set_xticklabels([])
    axs[0].set_ylabel("div [s$^{-1}$]")
    axs[1].set_ylabel("$\Delta$ div [s$^{-2}$]")
    axs[2].set_ylabel("thrust [g]")
    axs[2].set_xlabel("time [s]")
    # ax.set_xlabel("time [s]")
    for ax in axs:
        ax.grid()
        ax.set_facecolor("black")
        ax.set_facecolor("black")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    (line1,) = axs[0].plot(
        [],
        [],
        "r",
        # marker="^",
        # markersize=15,
        # markeredgecolor="white",
        # markerfacecolor="red",
        # linestyle=None,
        lw=2,
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
        lw=2,
    )
    (line3,) = axs[2].plot(
        [],
        [],
        "r",
        # marker="^",
        # markersize=15,
        # markeredgecolor="white",
        # markerfacecolor="red",
        # linestyle=None,
        lw=2,
    )

    # Create partial functions
    animate_p = partial(animate, data, line1, line2, line3)
    init_p = partial(init, line1, line2, line3)

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
        # fps=1 / (data["time"][len(data) - 1] / (len(data) - 1)),
        fps=fps,
        extra_args=["-vcodec", "libx264"],
        # savefig_kwargs={"facecolor": "#abd7e6"},
        savefig_kwargs={"facecolor": "black"},
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--fps", type=int, required=True)
    args = vars(parser.parse_args())

    animate_run(args["inp"], args["out"], args["fps"])
