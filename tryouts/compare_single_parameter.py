import argparse
from pathlib import Path

import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evolutionary.utils.constructors import build_network


def compare_single_parameter(folders, analyses, parameter, filter=False, pareto=False):
    folders = [Path(f) for f in folders]
    analyses = [Path(a) for a in analyses]

    # Glob all network files in subfolders
    files = [sorted(f.rglob("*.net")) for f in folders]

    # Optional (Pareto) filter
    if filter:
        if pareto:
            filters = [np.load(a / "mask_pareto.npy") for a in analyses]
            files = [np.array(f)[fil].tolist() for f, fil in zip(files, filters)]
        else:
            filters = [np.load(a / "mask.npy") for a in analyses]
            files = [np.array(f)[fil].tolist() for f, fil in zip(files, filters)]

    # Build network placeholders
    networks = []
    for f in folders:
        with open(f / "config.yaml", "r") as cf:
            config = yaml.full_load(cf)
            networks.append(build_network(config))

    # Dicts to hold everything in an orderly manner
    values = {i: {} for i in range(len(files))}

    # Go over networks
    for i, file, network in zip(range(len(files)), files, networks):
        for f in file:
            # Load parameters
            network.load_state_dict(torch.load(f))
            network.reset_state()

            # Add values to dict
            for name, child in network.named_children():
                if name not in values[i] and hasattr(child, parameter):
                    values[i][name] = []
                if hasattr(child, parameter):
                    values[i][name].append(
                        getattr(child, parameter).detach().clone().view(1, -1)
                    )

    # Convert to single numpy arrays
    for i, case in values.items():
        for layer, val in case.items():
            values[i][layer] = torch.cat(val, 0).view(-1).numpy()

    fig, axs = plt.subplots(len(files), 3, sharey=True, sharex=True)

    # Bins
    bins = np.linspace(0.0, 1.0, 15)

    for (i, case), analysis in zip(values.items(), analyses):
        for j, (layer, val) in zip(range(len(case.keys())), case.items()):
            axs[i, j].set_title(layer)
            axs[i, j].grid(True)
            axs[i, j].hist(
                val, bins, density=True, edgecolor="k", alpha=0.5, label=i, zorder=10
            )
            data = pd.DataFrame({"values": val})
            data.to_csv(
                analysis / f"parameters+{parameter}+{layer}.csv", index=False, sep=","
            )
        axs[i, 0].legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--analyses", nargs="+", required=True)
    parser.add_argument("--parameter", type=str, required=True)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--pareto", action="store_true")
    args = vars(parser.parse_args())

    # Call
    compare_single_parameter(
        args["folders"],
        args["analyses"],
        args["parameter"],
        filter=args["filter"],
        pareto=args["pareto"],
    )
