import argparse
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation

# Initialization function: plot the background of each frame
def init(line):
    line.set_data([], [])
    return (line,)


# Animation function, this is called sequentially
def animate(data, line, i):
    x = 0
    y = data["pos_z"][i]
    line.set_data(x, y)
    return (line,)


# Main function
def animate_run(inp, out):
    data = pd.read_csv(inp, sep=",")

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(2, 5), dpi=400)
    # fig.patch.set_facecolor("#abd7e6")
    fig.patch.set_facecolor("black")
    # fig.patch.set_alpha(0.0)
    ax = plt.axes(xlim=(-0.1, 0.1), ylim=(0, 5))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_ylabel("h [m]")
    ax.set_facecolor("black")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")

    (line,) = ax.plot(
        [],
        [],
        marker="^",
        markersize=15,
        markeredgecolor="white",
        markerfacecolor="red",
        linestyle=None,
        lw=2,
    )

    # Create partial functions
    animate_p = partial(animate, data, line)
    init_p = partial(init, line)

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
    anim.save(
        out,
        fps=1 / (data["time"][len(data) - 1] / (len(data) - 1)),
        extra_args=["-vcodec", "libx264"],
        # savefig_kwargs={"facecolor": "#abd7e6"},
        savefig_kwargs={"facecolor": "black"},
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = vars(parser.parse_args())

    animate_run(args["inp"], args["out"])
