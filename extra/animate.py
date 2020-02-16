import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation

data = pd.read_csv("run.csv", sep=",")

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(2, 5), dpi=400)
fig.patch.set_facecolor("#abd7e6")
# fig.patch.set_facecolor("none")
# fig.patch.set_alpha(0.0)
ax = plt.axes(xlim=(-0.1, 0.1), ylim=(0, 5))
ax.grid()
ax.set_xticklabels([])
ax.set_ylabel("h [m]")
# ax.set_facecolor('#abd7e6')
(line,) = ax.plot(
    [],
    [],
    marker="^",
    markersize=15,
    markeredgecolor="k",
    markerfacecolor="orange",
    linestyle=None,
    lw=2,
)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)


# animation function.  This is called sequentially
def animate(i):
    x = 0
    y = data["pos_z"][i]
    line.set_data(x, y)
    return (line,)


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(data), interval=0, blit=True
)

fig.tight_layout()
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save(
    "anim8.mp4",
    fps=1 / (data["time"][len(data) - 1] / (len(data) - 1)),
    extra_args=["-vcodec", "libx264"],
    savefig_kwargs={"facecolor": "#abd7e6"},
)

plt.show()
