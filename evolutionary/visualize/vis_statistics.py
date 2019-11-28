import pandas as pd
import matplotlib.pyplot as plt


def vis_statistics(config, parameters, verbose=2):
    # Load fitnesses
    with open(parameters + "fitnesses.txt", "r") as f:
        fitnesses = pd.read_csv(f, sep="\t")
    # Load sensitivity
    with open(config["log location"] + "sensitivity_stds.txt", "r") as f:
        sensitivity_stds = pd.read_csv(f, sep="\t")

    # Do some statistics
    # Correlation between number of spikes and all of the performance metrics
    # Hypothesis: less spikes = more variability
    assert (
        "spikes" in fitnesses.columns
    ), "Needs a count of the total amount of spikes in the network"
    spike_corr = [
        fitnesses["spikes"].corr(sensitivity_stds[col])
        for col in ["time to land", "final height", "final velocity"]
    ]

    # Plot in figure
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), dpi=200)
    for i, ax, col in zip(range(len(axs)), axs, sensitivity_stds.columns):
        ax.set_xlabel("objective sigma")
        ax.set_ylabel("avg # of spikes per second")
        ax.set_title(f"corr: {spike_corr[i]:.2f}")
        ax.grid()
        for i in range(fitnesses.shape[0]):
            ax.text(
                sensitivity_stds[col][i],
                fitnesses["spikes"][i],
                str(i),
                va="top",
                fontsize=7,
            )
        ax.scatter(sensitivity_stds[col], fitnesses["spikes"], s=6, label=col)
        ax.legend()
    fig.tight_layout()

    # Save figure
    if verbose:
        fig.savefig(f"{config['log location']}spike_correlation.png")

    # Show figure
    if verbose > 1:
        plt.show()
