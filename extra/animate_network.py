import argparse
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation

# Initialization function: plot the background of each frame
def init(verts):
    return (verts,)


# Animation function, this is called sequentially
def animate(data, verts, i):
    spikes = data.filter(regex=r"n\d+_spike").loc[i, :].to_numpy()
    spikes = np.concatenate(([0, 0, 0, 0], spikes), axis=0)
    colors = np.where(spikes == 0, "black", "red")
    verts.set_facecolors(colors)
    return (verts,)


# Main function
def animate_net(run, net, out):
    network = pd.read_csv(net, sep=",")
    data = pd.read_csv(run, sep=",")

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(1, 1.3), dpi=1000)
    fig.patch.set_facecolor("black")
    ax = plt.axes([0, 0, 1, 1], xlim=(-0.5, 2.5), ylim=(-2.8, 2.8))
    # ax = plt.axes(xlim=(-0.5, 4.5), ylim=(-2.8, 2.8))
    # ax.set_axes_off()
    plt.axis("off")
    verts = ax.scatter(
        network["x"], network["y"], c="black", s=20, edgecolors="white", linewidths=0.5
    )
    # (verts,) = ax.plot(
    #     network["x"],
    #     network["y"],
    #     marker="o",
    #     markersize=15,
    #     markeredgecolor="k",
    #     markerfacecolor="#abd7e6",
    #     linestyle=None,
    #     lw=2,
    # )

    # Create partial functions
    animate_p = partial(animate, data, verts)
    init_p = partial(init, verts)

    # Call the animator; blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate_p, init_func=init_p, frames=len(data), interval=0, blit=True
    )

    # fig.tight_layout()
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(
        out,
        fps=1 / (data["time"][len(data) - 1] / (len(data) - 1)),
        extra_args=["-vcodec", "libx264"],
        savefig_kwargs={"facecolor": "black"},
        # savefig_kwargs={"facecolor": "none"},
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = vars(parser.parse_args())

    animate_net(args["inp"], args["net"], args["out"])
