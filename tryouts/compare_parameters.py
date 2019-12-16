import argparse
from pathlib import Path

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pysnn.neuron import AdaptiveLIFNeuron

from evolutionary.utils.constructors import build_network


def compare_parameters(folder1, folder2, filter=False, pareto=False):
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    # Glob all network files in subfolders
    files1 = sorted(folder1.rglob("*.net"))
    files2 = sorted(folder2.rglob("*.net"))

    # Optional (Pareto) filter
    if filter:
        if pareto:
            filter1 = np.load(folder1 / "mask_pareto.npy")
            filter2 = np.load(folder2 / "mask_pareto.npy")
            files1 = np.array(files1)[filter1].tolist()
            files2 = np.array(files2)[filter2].tolist()
        else:
            filter1 = np.load(folder1 / "mask.npy")
            filter2 = np.load(folder2 / "mask.npy")
            files1 = np.array(files1)[filter1].tolist()
            files2 = np.array(files2)[filter2].tolist()

    # Genes we're going to compare
    genes = [
        "weight",
        "alpha_v",
        "alpha_t",
        "alpha_thresh",
        "tau_v",
        "tau_t",
        "tau_thresh",
        "thresh",
    ]

    # Build network placeholders
    with open(folder1 / "config.yaml", "r") as cf:
        config1 = yaml.full_load(cf)
    with open(folder2 / "config.yaml", "r") as cf:
        config2 = yaml.full_load(cf)
    network1 = build_network(config1)
    network2 = build_network(config2)

    # Dicts to hold everything in an orderly manner
    params1 = {gene: {} for gene in genes}
    params2 = {gene: {} for gene in genes}

    ### 1 ###
    # Go over networks
    for file in files1:
        # Load parameters
        network1.load_state_dict(torch.load(file))
        network1.reset_state()

        # Add values to dict
        for gene in genes:
            for name, child in network1.named_children():
                if name not in params1[gene] and hasattr(child, gene):
                    params1[gene][name] = []
                if hasattr(child, gene):
                    params1[gene][name].append(
                        getattr(child, gene).detach().clone().view(1, -1)
                    )

    # Remove unwanted ones, such as thresh for AdaptiveLIFNeuron
    for name, child in network1.named_children():
        if isinstance(child, AdaptiveLIFNeuron):
            params1["thresh"].pop(name, None)

    # Convert to single numpy arrays
    for gene, layers in params1.items():
        for layer, values in layers.items():
            params1[gene][layer] = torch.cat(values, 0).view(-1).numpy()

    ### 2 ###
    # Go over networks
    for file in files2:
        # Load parameters
        network2.load_state_dict(torch.load(file))
        network2.reset_state()

        # Add values to dict
        for gene in genes:
            for name, child in network2.named_children():
                if name not in params2[gene] and hasattr(child, gene):
                    params2[gene][name] = []
                if hasattr(child, gene):
                    params2[gene][name].append(
                        getattr(child, gene).detach().clone().view(1, -1)
                    )

    # Remove unwanted ones, such as thresh for AdaptiveLIFNeuron
    for name, child in network2.named_children():
        if isinstance(child, AdaptiveLIFNeuron):
            params2["thresh"].pop(name, None)

    # Convert to single numpy arrays
    for gene, layers in params2.items():
        for layer, values in layers.items():
            params2[gene][layer] = torch.cat(values, 0).view(-1).numpy()

    # Plot for each gene
    for gene in genes:
        fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)

        # Compute overall min/max for a certain gene
        min_gene = min(
            [
                l.min().item()
                for params in [params1, params2]
                for g, layers in params.items()
                for l in layers.values()
                if g == gene
            ]
        )
        max_gene = max(
            [
                l.max().item()
                for params in [params1, params2]
                for g, layers in params.items()
                for l in layers.values()
                if g == gene
            ]
        )
        bins = np.linspace(min_gene, max_gene, 15)

        for ax, (layer, values1), values2 in zip(
            axs, params1[gene].items(), params2[gene].values()
        ):
            ax.set_title(f"{gene}: {layer}")
            ax.grid()
            ax.hist(
                values1,
                bins,
                density=True,
                edgecolor="k",
                alpha=0.5,
                label="1",
                zorder=10,
            )
            ax.hist(
                values2,
                bins,
                density=True,
                edgecolor="k",
                alpha=0.5,
                label="2",
                zorder=100,
            )
        axs[0].legend()
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--pareto", action="store_true")
    args = vars(parser.parse_args())

    # Call
    compare_parameters(
        args["folder1"], args["folder2"], filter=args["filter"], pareto=args["pareto"]
    )
