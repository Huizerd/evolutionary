import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_relevant(population, hof, obj_labels, plot_obj, last=None, verbose=2):
    # Set indices and limits based on labels
    assert len(obj_labels) >= 3, "Only 3 or more objectives are supported"
    assert len(plot_obj) == 2, "Provide 2 dimensions/objectives to be plotted"
    obj_limits = []
    for obj in obj_labels:
        if obj == "time to land":
            obj_limits.append(10.0)
        elif obj == "final velocity":
            obj_limits.append(2.0)
        elif obj == "spikes":
            obj_limits.append(200.0)
        else:
            obj_limits.append(np.inf)

    # Get relevant fitness values
    fitnesses = np.array(
        [list(ind.fitness.values) + [i] for i, ind in enumerate(population)]
    )
    fitnesses_hof = np.array(
        [list(ind.fitness.values) + [i] for i, ind in enumerate(hof)]
    )
    mask = (fitnesses[:, :-1] < np.array(obj_limits)).all(1)
    mask_hof = (fitnesses_hof[:, :-1] < np.array(obj_limits)).all(1)
    fitnesses = fitnesses[mask]
    fitnesses_hof = fitnesses_hof[mask_hof]

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
        ax.scatter(fitnesses[:, plot_obj[0]], fitnesses[:, plot_obj[1]])
    if fitnesses_hof.size > 0:
        ax.scatter(fitnesses_hof[:, plot_obj[0]], fitnesses_hof[:, plot_obj[1]])

    # Annotate
    for i in range(fitnesses.shape[0]):
        ax.text(
            fitnesses[i, plot_obj[0]],
            fitnesses[i, plot_obj[1]],
            str(int(fitnesses[i, -1])),
        )
    for i in range(fitnesses_hof.shape[0]):
        ax.text(
            fitnesses_hof[i, plot_obj[0]],
            fitnesses_hof[i, plot_obj[1]],
            str(int(fitnesses_hof[i, -1])),
            va="top",
        )

    # Decorate figure
    ax.set_xlabel(obj_labels[plot_obj[0]])
    ax.set_ylabel(obj_labels[plot_obj[1]])
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    if verbose > 1:
        fig.canvas.draw()
        fig.canvas.flush_events()

    return fig, ax, True


def vis_population(population, hof, obj_labels, plot_obj, last=None, verbose=2):
    # Create figure and axis if not there, else unpack
    assert len(plot_obj) == 3, "Provide 3 dimensions/objectives to be plotted"
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
    fitnesses_hof = np.array([ind.fitness.values for ind in hof])
    ax.scatter(
        fitnesses[:, plot_obj[0]], fitnesses[:, plot_obj[1]], fitnesses[:, plot_obj[2]]
    )
    ax.scatter(
        fitnesses_hof[:, plot_obj[0]],
        fitnesses_hof[:, plot_obj[1]],
        fitnesses_hof[:, plot_obj[2]],
    )

    # Annotate only population for now
    for i in range(fitnesses.shape[0]):
        ax.text(
            fitnesses[i, plot_obj[0]],
            fitnesses[i, plot_obj[1]],
            fitnesses[i, plot_obj[2]],
            str(i),
        )
    for i in range(fitnesses_hof.shape[0]):
        ax.text(
            fitnesses_hof[i, plot_obj[0]],
            fitnesses_hof[i, plot_obj[1]],
            fitnesses_hof[i, plot_obj[2]],
            str(i),
            va="top",
        )

    # Decorate figure
    ax.set_xlabel(obj_labels[plot_obj[0]])
    ax.set_ylabel(obj_labels[plot_obj[1]])
    ax.set_zlabel(obj_labels[plot_obj[2]])
    ax.grid()

    # Update/draw figure
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    if verbose > 1:
        fig.canvas.draw()

    return fig, ax
