import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_population(population, hof, last=None, verbose=2):
    # Get relevant fitness values
    fitnesses = np.array(
        [list(ind.fitness.values) + [i] for i, ind in enumerate(population)]
    )
    fitnesses_hof = np.array(
        [list(ind.fitness.values) + [i] for i, ind in enumerate(hof)]
    )

    # Create figure and axis if not there, else unpack or leave
    # We had no figure and want one
    if last is None:
        plt.ion()
        fig, ax = plt.subplots(1, 1, dpi=200)
    # We had a figure
    else:
        fig, ax, _ = last

    # We (still) don't have anything to plot
    if fitnesses.size == 0 and fitnesses_hof.size == 0:
        return fig, ax, False

    # Clear axis
    ax.cla()

    # Plot the fitnesses
    if fitnesses.size > 0:
        ax.scatter(fitnesses[:, 0], np.zeros_like(fitnesses[:, 0]))
    if fitnesses_hof.size > 0:
        ax.scatter(fitnesses_hof[:, 0], np.zeros_like(fitnesses_hof[:, 0]))

    # Annotate
    for i in range(fitnesses.shape[0]):
        ax.text(
            fitnesses[i, 0], 0, str(int(fitnesses[i, -1])),
        )
    for i in range(fitnesses_hof.shape[0]):
        ax.text(
            fitnesses_hof[i, 0], 0, str(int(fitnesses_hof[i, -1])), va="top",
        )

    # Decorate figure
    ax.set_xlabel(f"SSE D {population[0][0].setpoint}")
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    if verbose > 1:
        fig.canvas.draw()
        fig.canvas.flush_events()

    return fig, ax, True
