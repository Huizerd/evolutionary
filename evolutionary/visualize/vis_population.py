import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_relevant(population, obj_idx, obj_labels, last=None):
    # Create figure and axis if not there, else unpack
    if last is None:
        plt.ion()
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111)
    else:
        fig, ax = last

    # Clear axis
    ax.cla()

    # Get relevant fitness values and plot
    fitnesses = np.array(
        [
            ind.fitness.values + (i,)
            for i, ind in enumerate(population)
            if ind.fitness.values[obj_idx[0][0]] < obj_idx[0][1]
            and ind.fitness.values[obj_idx[1][0]] < obj_idx[1][1]
        ]
    )
    if fitnesses.size > 0:
        ax.scatter(fitnesses[:, obj_idx[0][0]], fitnesses[:, obj_idx[1][0]])
    else:
        return fig, ax

    # Annotate
    for i in range(fitnesses.shape[0]):
        ax.text(
            fitnesses[i, obj_idx[0][0]],
            fitnesses[i, obj_idx[1][0]],
            str(int(fitnesses[i, 3])),
        )

    # Decorate figure
    ax.set_xlabel(obj_labels[obj_idx[0][0]])
    ax.set_ylabel(obj_labels[obj_idx[1][0]])
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax


def vis_population(population, obj_labels, last=None):
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

    # Annotate
    for i in range(fitnesses.shape[0]):
        ax.text(fitnesses[i, 0], fitnesses[i, 1], fitnesses[i, 2], str(i))

    # Decorate figure
    ax.set_xlabel(obj_labels[0])
    ax.set_ylabel(obj_labels[1])
    ax.set_zlabel(obj_labels[2])
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax
