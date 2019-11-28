import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from evolutionary.utils.constructors import build_network


def vis_out_dynamics(config, parameters, verbose=2):
    # Expand to all parameter files
    parameters = sorted(glob.glob(parameters + "*.net"))

    # Build network
    network = build_network(config)

    # For each set of parameters, add the alpha and tau for the voltage/trace of the final neuron
    dynamics = np.zeros((len(parameters), 4))

    for i, param in enumerate(parameters):
        # Load network
        network.load_state_dict(torch.load(param))
        # Write to array
        dynamics[i, :] = [
            network.neuron2.alpha_v.item(),
            network.neuron2.tau_v.item(),
            network.neuron2.alpha_t.item(),
            network.neuron2.tau_t.item(),
        ]

    # Plot voltage in figure
    fig_v, ax_v = plt.subplots(1, 1, dpi=200)
    ax_v.set_title("Output dynamics: voltage")
    ax_v.set_xlabel("alpha")
    ax_v.set_ylabel("tau")
    ax_v.grid()
    for i in range(dynamics.shape[0]):
        ax_v.text(dynamics[i, 0], dynamics[i, 1], str(i), va="top", fontsize=7)
    ax_v.scatter(dynamics[:, 0], dynamics[:, 1], s=6)
    fig_v.tight_layout()

    # Save figure
    if verbose:
        fig_v.savefig(f"{config['log location']}out_dynamics_volt.png")

    # Plot trace in figure
    fig_t, ax_t = plt.subplots(1, 1, dpi=200)
    ax_t.set_title("Output dynamics: trace")
    ax_t.set_xlabel("alpha")
    ax_t.set_ylabel("tau")
    ax_t.grid()
    for i in range(dynamics.shape[0]):
        ax_t.text(dynamics[i, 2], dynamics[i, 3], str(i), va="top", fontsize=7)
    ax_t.scatter(dynamics[:, 2], dynamics[:, 3], s=6)
    fig_t.tight_layout()

    # Save figure
    if verbose:
        fig_t.savefig(f"{config['log location']}out_dynamics_trace.png")

    # Show figures
    if verbose > 1:
        plt.show()
