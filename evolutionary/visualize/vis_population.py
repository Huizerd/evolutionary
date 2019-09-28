import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d


def vis_population(population, last=None):
    # Create figure and axis if not there, else unpack
    if last is None:
        plt.ion()
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = last

    # Clear axis
    ax.cla()

    # Get all fitness values and plot
    fitnesses = np.array([ind.fitness.values for ind in population])
    ax.scatter(fitnesses[:, 0], fitnesses[:, 1], fitnesses[:, 2])

    # Decorate figure
    ax.set_xlabel("Time to land")
    ax.set_ylabel("Final altitude")
    ax.set_zlabel("Final velocity")
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax
