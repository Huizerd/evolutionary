from collections import OrderedDict

import torch
import matplotlib.pyplot as plt

from evolutionary.utils.constructors import build_network


def vis_network(config, parameters, debug=False, no_plot=False):
    # Load network
    network = build_network(config)
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
            elif gene == "bias":
                im = axs[0, i].imshow(
                    param.view(1, -1).numpy(),
                    cmap="plasma",
                    vmin=param_min,
                    vmax=param_max,
                )
                axs[0, i].set_xlabel("post neuron id")
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

    if not debug and not no_plot:
        plt.show()


def vis_distributions(config, parameters, debug=False, no_plot=False):
    assert len(parameters) > 1, "Include more than one network file for analysis"

    # Get empty network, and fill with first set of parameters (needed for init)
    network = build_network(config)
    network.load_state_dict(torch.load(parameters[0]))

    # Go over all genes and create separate figures
    # Note that weights/delays belong to connections/layers, while others belong to neurons
    for gene in config["evo"]["genes"]:
        # Dict of tensors that we can average over
        params = OrderedDict(
            [
                (name, getattr(child, gene).detach().clone()[None])
                for name, child in network.named_children()
                if hasattr(child, gene)
            ]
        )

        for p in parameters[1:]:
            # Get next set of parameters
            network.load_state_dict(torch.load(p))
            for name, tens in params.items():
                for name2, child in network.named_children():
                    if name == name2 and hasattr(child, gene):
                        params[name] = torch.cat(
                            [tens, getattr(child, gene).detach().clone()[None]], 0
                        )

        # Average over first dim
        params_avg = OrderedDict(
            [(name, tens.mean(0)) for name, tens in params.items()]
        )

        # Get overall min and max for color bar
        param_min = min([p.min().item() for p in params.values()])
        param_max = max([p.max().item() for p in params.values()])

        # Build figure
        fig, axs = plt.subplots(2, len(params))
        for i, (name, param), (_, param_avg) in zip(
            range(axs.shape[1]), params.items(), params_avg.items()
        ):
            # Colored value plots
            if gene == "weight":
                im = axs[0, i].imshow(
                    param_avg.T.numpy(), cmap="plasma", vmin=param_min, vmax=param_max
                )
                axs[0, i].set_xlabel("post neuron id")
                axs[0, i].set_ylabel("pre neuron id")
            elif gene == "bias":
                im = axs[0, i].imshow(
                    param_avg.view(1, -1).numpy(),
                    cmap="plasma",
                    vmin=param_min,
                    vmax=param_max,
                )
                axs[0, i].set_xlabel("post neuron id")
            else:
                im = axs[0, i].imshow(
                    param_avg.numpy(), cmap="plasma", vmin=param_min, vmax=param_max
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
                f"{config['log location']}{gene}+distribution+{'_'.join(config['individual id'])}.png"
            )

    if not debug and not no_plot:
        plt.show()
