from collections import OrderedDict

import torch
import matplotlib.pyplot as plt

from pysnn.connection import Connection

from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN


def vis_network(config, parameters):
    # Load network
    if config["network"] == "ANN":
        network = ANN(2, config["hidden size"], 1)
    elif config["network"] == "SNN":
        network = SNN(2, config["hidden size"], 1)
    else:
        raise KeyError("Not a valid network key!")

    network.load_state_dict(torch.load(parameters))

    # Collect weights based on module being a connection or not
    weights = OrderedDict(
        [
            (name, module.weight.T.clone())
            for name, module in network.named_modules()
            if isinstance(module, Connection)
        ]
    )
    w_min = min([w.min() for w in weights.values()])
    w_max = max([w.max() for w in weights.values()])

    # Build figure
    fig, axs = plt.subplots(1, len(weights))
    for ax, (name, weight) in zip(axs, weights.items()):
        im = ax.imshow(weight.numpy(), cmap="plasma", vmin=w_min, vmax=w_max)
        ax.set_title(name)
        ax.set_xlabel("Postsynaptic neuron ID")
        ax.set_ylabel("Presynaptic neuron ID")

    fig.colorbar(im, ax=axs[-1], orientation="vertical", fraction=0.1)
    fig.tight_layout()
    plt.show()
