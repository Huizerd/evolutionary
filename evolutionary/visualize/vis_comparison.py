import glob
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import numpy as np

from pysnn.network import SNNNetwork

from evolutionary.environment.environment import QuadEnv
from evolutionary.utils.constructors import build_network


def vis_comparison(configs, comparison, verbose=2):
    # We compare in terms of performance and genes/parameters
    genes = [
        "weight",
        "bias",
        "alpha_v",
        "alpha_t",
        "alpha_thresh",
        "tau_v",
        "tau_t",
        "tau_thresh",
    ]
    performance = OrderedDict()
    values = OrderedDict()
    statics = OrderedDict()

    # Go over groups
    assert len(configs) == len(
        comparison["parameters"]
    ), "Exactly one config file per parameter set"
    for config, parameters, name in zip(
        configs, comparison["parameters"], comparison["names"]
    ):
        # Expand to multiple files if needed
        parameters = glob.glob(parameters)

        # Build network
        network = build_network(config)

        # Build environment
        env = QuadEnv(
            delay=config["env"]["delay"][0],
            noise=config["env"]["noise"][0],
            noise_p=config["env"]["noise p"][0],
            thrust_bounds=config["env"]["thrust bounds"],
            thrust_tc=config["env"]["thrust tc"][0],
            settle=config["env"]["settle"],
            wind=config["env"]["wind"],
            h0=config["env"]["h0"][0],
            dt=config["env"]["dt"],
            max_t=config["env"]["max time"],
            seed=None,
        )

        # Add subdicts
        performance[name] = OrderedDict()
        values[name] = OrderedDict()
        statics[name] = OrderedDict([("dt", config["env"]["dt"])])

        # Get all data/results
        for id, param in enumerate(parameters):
            # Subdict for each parameter set
            performance[name][id] = OrderedDict()

            # Load network parameters
            network.load_state_dict(torch.load(param))

            # Add values to dict
            for gene in genes:
                if gene in values[name]:
                    for layer, tens in values[name][gene].items():
                        for layer2, child in network.named_children():
                            if layer == layer2 and hasattr(child, gene):
                                values[name][gene][layer] = torch.cat(
                                    [tens, getattr(child, gene).detach().clone()[None]],
                                    0,
                                )
                else:
                    values[name][gene] = OrderedDict(
                        [
                            (n, getattr(child, gene).detach().clone()[None])
                            for n, child in network.named_children()
                            if hasattr(child, gene)
                        ]
                    )

                # Get overall min and max
                try:
                    statics[gene] = (
                        min(
                            [
                                p.min().item()
                                for n in values.values()
                                for p in n[gene].values()
                            ]
                        ),
                        max(
                            [
                                p.max().item()
                                for n in values.values()
                                for p in n[gene].values()
                            ]
                        ),
                    )
                except (ValueError, TypeError):
                    statics[gene] = None

            # Test performance
            for h in config["env"]["h0"]:
                performance[name][id][h] = [[] for _ in range(5)]
                for i in range(len(performance[name][id][h])):
                    if isinstance(network, SNNNetwork):
                        network.reset_state()
                    obs = env.reset(h0=h)
                    done = False

                    while not done:
                        # Log performance
                        performance[name][id][h][i].append(env.state.copy())

                        # Step the environment
                        obs = torch.from_numpy(obs)
                        action = network.forward(obs.view(1, 1, -1))
                        action = action.numpy()
                        obs, _, done, _ = env.step(action)

                    # Convert to numpy array
                    performance[name][id][h][i] = np.array(performance[name][id][h][i])

    # First plot performance
    colors = ["xkcd:neon red", "xkcd:neon blue", "xkcd:neon green", "xkcd:neon purple"]
    assert len(colors) >= len(configs), "Add more colors!"
    fig_p, axs_p = plt.subplots(3, 4, figsize=(20, 10), sharex="col", sharey="row")
    for i, (name, params) in enumerate(performance.items()):
        for id, heights in params.items():
            for k, height in enumerate(heights.values()):
                axs_p[-1, k].set_xlabel("time [s]")
                for l, run in enumerate(height):
                    for m, var in enumerate(
                        ["height [m]", "velocity [m/s]", "acceleration [m/s2]"]
                    ):
                        axs_p[m, 0].set_ylabel(var)
                        label = name if id == 0 and l == 0 else ""
                        axs_p[m, k].plot(
                            np.linspace(
                                0.0,
                                (run.shape[0] - 1) * statics[name]["dt"],
                                run.shape[0],
                            ),
                            run[:, m],
                            color=colors[i],
                            label=label,
                        )
                        axs_p[m, k].legend()
                        axs_p[m, k].grid(True)
    fig_p.tight_layout()

    if verbose:
        fig_p.savefig(f"{comparison['log location']}performance.png")

    # Then plot distributions of network parameters
    for gene in genes:
        if statics[gene] is None:
            continue
        fig_g, axs_g = plt.subplots(
            len(configs), 3, sharey=True, sharex=True, squeeze=False
        )
        for i, (name, params) in enumerate(values.items()):
            axs_g[i, 0].set_ylabel("count")
            for j, (layer, tens) in enumerate(params[gene].items()):
                axs_g[i, j].set_xlabel(f"{gene} value")
                axs_g[i, j].hist(
                    tens.view(-1).numpy(),
                    range=statics[gene],
                    color=colors[i],
                    label=name,
                )
                axs_g[i, j].set_title(f"{gene}: {layer}")
                axs_g[i, j].grid(True)
                axs_g[i, j].legend()
        fig_g.tight_layout()

        if verbose:
            fig_g.savefig(f"{comparison['log location']}{gene}.png")

    if verbose > 1:
        plt.show()
