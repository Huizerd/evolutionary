from collections import OrderedDict

import torch
import matplotlib.pyplot as plt

from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN


def vis_network(config, parameters, debug=False, no_plot=False):
    # Load network
    if config["network"] == "ANN":
        network = ANN(2, config["hidden size"], 1)
    elif config["network"] == "SNN":
        if config["double neurons"]:
            inputs = 4
        else:
            inputs = 2
        if config["double actions"]:
            outputs = 2
        else:
            outputs = 1
        network = SNN(inputs, config["hidden size"], outputs, config)
    else:
        raise KeyError("Not a valid network key!")

    network.load_state_dict(torch.load(parameters))

    # Go over all genes and create separate figures
    # Note that weights/delays belong to connections/layers, while others belong to neurons
    for gene in config["evo"]["genes"]:
        params = OrderedDict(
            [
                (name, getattr(child, gene).detach().clone())
                for name, child in network.named_children()
                if hasattr(child, gene)
            ]
        )
        param_min = min([p.min().item() for p in params.values()])
        param_max = max([p.max().item() for p in params.values()])

        # Build figure
        fig, axs = plt.subplots(2, len(params))
        for i, (name, param) in zip(range(axs.shape[1]), params.items()):
            # Colored value plots
            if gene == "weight":
                im = axs[0, i].imshow(
                    param.T.numpy(), cmap="plasma", vmin=param_min, vmax=param_max
                )
                axs[0, i].set_xlabel("post neuron id")
                axs[0, i].set_ylabel("pre neuron id")
            else:
                im = axs[0, i].imshow(
                    param.numpy(), cmap="plasma", vmin=param_min, vmax=param_max
                )
            axs[0, i].set_title(f"{gene}: {name}")
            fig.colorbar(im, ax=axs[0, i], orientation="vertical", fraction=0.1)
            axs[0, i].tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )

            # Histograms
            axs[1, i].hist(param.view(-1).numpy(), range=(param_min, param_max))
            axs[1, i].set_xlabel(f"{gene} value")
            axs[1, i].set_ylabel("count")
            axs[1, i].grid()

        fig.tight_layout()

        if not debug:
            fig.savefig(
                f"{config['log location']}{gene}+{'_'.join(config['individual id'])}.png"
            )

    if not no_plot:
        plt.show()
