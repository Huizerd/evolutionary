from collections import OrderedDict

import torch
import yaml
import matplotlib.pyplot as plt

from evolutionary.utils.constructors import build_network
from evolutionary.visualize.colormap import parula_map


def vis_network(config, parameters, verbose=2):
    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

    # Get parameters that are more suited to be printed in text
    params = {"evolved": {}, "fixed": {}}
    for param in ["alpha_v", "alpha_t", "alpha_thresh", "tau_v", "tau_t", "tau_thresh"]:
        values = OrderedDict(
            [
                (name, round(getattr(child, param).item(), 3))
                for name, child in network.named_children()
                if hasattr(child, param)
            ]
        )
        if param in config["evo"]["genes"]:
            params["evolved"][param] = values
        else:
            params["fixed"][param] = values

    # Print parameters to file
    if verbose:
        with open(config["log location"] + "net_param.yaml", "w") as f:
            yaml.dump(params, f, default_flow_style=False)

    # Go over all genes and create separate figures
    # Note that weights/delays belong to connections/layers, while others belong to neurons
    for gene in config["evo"]["genes"]:
        if gene not in ["weight", "bias", "thresh"]:
            continue

        # Get parameters
        params = OrderedDict(
            [
                (name, getattr(child, gene).detach().clone())
                for name, child in network.named_children()
                if hasattr(child, gene)
            ]
        )
        param_min = min([p.min().item() for p in params.values()])
        param_max = max([p.max().item() for p in params.values()])

        # Build figure if applicable
        fig, axs = plt.subplots(2, len(params), squeeze=False)
        for i, (name, param) in zip(range(axs.shape[1]), params.items()):
            # Colored value plots
            if gene == "weight":
                im = axs[0, i].imshow(
                    param.T.numpy(), cmap=parula_map, vmin=param_min, vmax=param_max
                )
                axs[0, i].set_xlabel("post neuron id")
                axs[0, i].set_ylabel("pre neuron id")
            elif gene == "thresh" or gene == "bias":
                im = axs[0, i].imshow(
                    param.view(1, -1).numpy(),
                    cmap=parula_map,
                    vmin=param_min,
                    vmax=param_max,
                )
                axs[0, i].set_xlabel("post neuron id")

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

        if verbose:
            fig.savefig(f"{config['log location']}{gene}.png")

    if verbose > 1:
        plt.show()
