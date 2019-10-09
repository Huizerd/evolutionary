from collections import OrderedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pysnn.connection import Linear
from pysnn.neuron import Neuron

from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN


def vis_network(config, parameters):
    # Load network
    if config["network"] == "ANN":
        network = ANN(2, config["hidden size"], 1)
    elif config["network"] == "SNN":
        network = SNN(2, config["hidden size"], 1, config)
    else:
        raise KeyError("Not a valid network key!")

    network.load_state_dict(torch.load(parameters))

    # Collect weights based on module being a connection or not
    weights = OrderedDict(
        [
            (name, child.weight.T.detach().clone())
            for name, child in network.named_children()
            if isinstance(child, Linear) or isinstance(child, nn.Linear)
        ]
    )
    w_min = min([w.min() for w in weights.values()])
    w_max = max([w.max() for w in weights.values()])

    # Collect thresholds for neurons that have them
    thresholds = OrderedDict(
        [
            (name, child.thresh.detach().clone().view(-1, 1))
            for name, child in network.named_children()
            if isinstance(child, Neuron) and "thresh" in config["genes"]
        ]
    )
    thresh_min = min([t.min() for t in thresholds.values()])
    thresh_max = max([t.max() for t in thresholds.values()])

    # Build weight figure
    fig_w, axs_w = plt.subplots(1, len(weights))
    for ax, (name, weight) in zip(axs_w, weights.items()):
        im = ax.imshow(weight.numpy(), cmap="plasma", vmin=w_min, vmax=w_max)
        ax.set_title(name)
        ax.set_xlabel("postsynaptic neuron id")
        ax.set_ylabel("presynaptic neuron id")

    fig_w.colorbar(im, ax=axs_w[-1], orientation="vertical", fraction=0.1)
    fig_w.tight_layout()
    fig_w.savefig(
        f"{config['log location']}weights+{'_'.join(config['individual id'])}.png"
    )

    # Build thresh figure if applicable
    if thresholds:
        fig_t, axs_t = plt.subplots(1, len(thresholds))
        for ax, (name, thresh) in zip(axs_t, thresholds.items()):
            im = ax.imshow(
                thresh.numpy(), cmap="plasma", vmin=thresh_min, vmax=thresh_max
            )
            ax.set_title(name)
            ax.set_ylabel("neuron id")

        fig_t.colorbar(im, ax=axs_t[-1], orientation="vertical", fraction=0.1)
        fig_t.tight_layout()
        fig_t.savefig(
            f"{config['log location']}thresholds+{'_'.join(config['individual id'])}.png"
        )

    plt.show()
