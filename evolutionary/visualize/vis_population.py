import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_relevant(population, hof, obj_labels, runs, last=None, verbose=2):
    # Set indices and limits based on labels
    assert len(obj_labels) == 3, "Only 3 objectives are supported"
    obj_limits = []
    for obj in obj_labels:
        if obj == "unsigned divergence" or obj == "signed divergence":
            obj_limits.append(250.0)
        elif obj == "final offset" or obj == "final offset 5m":
            obj_limits.append(1.0)
        elif obj == "time to land":
            obj_limits.append(10.0)
        elif obj == "final velocity":
            obj_limits.append(3.0)
        elif obj == "final velocity linear":
            obj_limits.append(2.0)
        else:
            obj_limits.append(np.inf)

    # Get relevant fitness values
    fitnesses = np.array(
        [
            [v / runs for v in ind.fitness.values] + [i]
            for i, ind in enumerate(population)
            if all(
                [
                    f < lim
                    for f, lim in zip(
                        [v / runs for v in ind.fitness.values], obj_limits
                    )
                ]
            )
        ]
    )
    fitnesses_hof = np.array(
        [
            [v / runs for v in ind.fitness.values] + [i]
            for i, ind in enumerate(hof)
            if all(
                [
                    f < lim
                    for f, lim in zip(
                        [v / runs for v in ind.fitness.values], obj_limits
                    )
                ]
            )
        ]
    )

    # Create figure and axis if not there, else unpack or leave
    # We had no figure and want one
    if last is None and (fitnesses.size > 0 or fitnesses_hof.size > 0):
        plt.ion()
        fig, ax = plt.subplots(1, 1, dpi=200)
    # We had a figure and want to use the same
    elif last is not None and (fitnesses.size > 0 or fitnesses_hof.size > 0):
        fig, ax = last
    # We had a figure and don't want to update it
    elif last is not None and (fitnesses.size == 0 and fitnesses_hof.size == 0):
        return last
    # We had no figure and don't want one
    else:
        return None

    # Clear axis
    ax.cla()

    # Plot the fitnesses
    if fitnesses.size > 0:
        ax.scatter(fitnesses[:, 0], fitnesses[:, 2])
    if fitnesses_hof.size > 0:
        ax.scatter(fitnesses_hof[:, 0], fitnesses_hof[:, 2])

    # Annotate
    for i in range(fitnesses.shape[0]):
        ax.text(fitnesses[i, 0], fitnesses[i, 2], str(int(fitnesses[i, 3])))
    for i in range(fitnesses_hof.shape[0]):
        ax.text(
            fitnesses_hof[i, 0],
            fitnesses_hof[i, 2],
            str(int(fitnesses_hof[i, 3])),
            va="top",
        )

    # Decorate figure
    ax.set_xlabel(obj_labels[0])
    ax.set_ylabel(obj_labels[2])
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    if verbose == 2:
        fig.canvas.draw()
        fig.canvas.flush_events()

    return fig, ax


def vis_population(population, hof, obj_labels, runs, last=None, verbose=2):
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
    fitnesses = np.array([ind.fitness.values for ind in population]) / runs
    fitnesses_hof = np.array([ind.fitness.values for ind in hof]) / runs
    ax.scatter(fitnesses[:, 0], fitnesses[:, 1], fitnesses[:, 2])
    ax.scatter(fitnesses_hof[:, 0], fitnesses_hof[:, 1], fitnesses_hof[:, 2])

    # Annotate only population for now
    for i in range(fitnesses.shape[0]):
        ax.text(fitnesses[i, 0], fitnesses[i, 1], fitnesses[i, 2], str(i))
    for i in range(fitnesses_hof.shape[0]):
        ax.text(
            fitnesses_hof[i, 0],
            fitnesses_hof[i, 1],
            fitnesses_hof[i, 2],
            str(i),
            va="top",
        )

    # Decorate figure
    ax.set_xlabel(obj_labels[0])
    ax.set_ylabel(obj_labels[1])
    ax.set_zlabel(obj_labels[2])
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    if verbose == 2:
        fig.canvas.draw()
        fig.canvas.flush_events()

    return fig, ax
