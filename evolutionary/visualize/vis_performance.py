from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt

from pysnn.neuron import BaseNeuron
from pysnn.network import SNNNetwork

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env


def vis_performance(config, parameters, verbose=2):
    # Build environment
    env = build_environment(config)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

    # Go over all heights we trained for
    for h in config["env"]["h0"]:
        # Randomize env for each altitude
        env = randomize_env(env, config)
        # Reset network and env
        if isinstance(network, SNNNetwork):
            network.reset_state()
        obs = env.reset(h0=h)
        done = False

        # For plotting
        state_list = []
        obs_gt_list = []
        obs_list = []
        time_list = []
        encoding_list = []

        # For neuron visualization
        neuron_dict = OrderedDict(
            [
                (name, {"trace": [], "volt": [], "spike": [], "thresh": []})
                for name, child in network.named_children()
                if isinstance(child, BaseNeuron)
            ]
        )

        while not done:
            # Log performance
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())
            time_list.append(env.t)

            # Log neurons
            for name, child in network.named_children():
                if name in neuron_dict:
                    neuron_dict[name]["trace"].append(
                        child.trace.detach().clone().view(-1).numpy()
                    )
                    neuron_dict[name]["volt"].append(
                        child.v_cell.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "v_cell") else None
                    neuron_dict[name]["spike"].append(
                        child.spikes.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "spikes") else None
                    neuron_dict[name]["thresh"].append(
                        child.thresh.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "thresh") else None

            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)

            # Log encoding as well
            if isinstance(network, SNNNetwork):
                encoding_list.append(network.input.view(-1).numpy())

        # Plot
        fig_p, axs_p = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
        # Height
        axs_p[0].plot(time_list, np.array(state_list)[:, 0], label="Height")
        axs_p[0].set_ylabel("height [m]")
        # Velocity
        axs_p[1].plot(time_list, np.array(state_list)[:, 1], label="Velocity")
        axs_p[1].set_ylabel("velocity [m/s]")
        # Acceleration/thrust
        axs_p[2].plot(time_list, np.array(state_list)[:, 2], label="Thrust")
        axs_p[2].set_ylabel("acceleration [m/s2]")
        # Divergence
        axs_p[3].plot(time_list, np.array(obs_gt_list)[:, 0], label="GT divergence")
        axs_p[3].plot(time_list, np.array(obs_list)[:, 0], label="Divergence")
        if isinstance(network, SNNNetwork) and network.encoding == "both":
            axs_p[3].plot(time_list, np.array(encoding_list)[:, 0], label="Encoded +D")
            axs_p[3].plot(time_list, np.array(encoding_list)[:, 2], label="Encoded -D")
        elif isinstance(network, SNNNetwork) and network.encoding == "divergence":
            axs_p[3].plot(time_list, np.array(encoding_list)[:, 0], label="Encoded +D")
            axs_p[3].plot(time_list, np.array(encoding_list)[:, 1], label="Encoded -D")
        axs_p[3].set_ylabel("divergence [1/s]")
        # Divergence dot
        axs_p[4].plot(time_list, np.array(obs_gt_list)[:, 1], label="GT div dot")
        axs_p[4].plot(time_list, np.array(obs_list)[:, 1], label="Div dot")
        if isinstance(network, SNNNetwork) and network.encoding == "both":
            axs_p[4].plot(
                time_list, np.array(encoding_list)[:, 1], label="Encoded +Ddot"
            )
            axs_p[4].plot(
                time_list, np.array(encoding_list)[:, 3], label="Encoded -Ddot"
            )
        axs_p[4].set_ylabel("divergence dot [1/s2]")
        axs_p[4].set_xlabel("time [s]")

        for ax in axs_p:
            ax.grid()
            ax.legend()
        fig_p.tight_layout()

        if verbose:
            plt.savefig(f"{config['log location']}performance+{int(h)}m.png")

        # Plot neurons
        if isinstance(network, SNNNetwork):
            dpi = 50 if config["net"]["hidden size"] > 10 else 100
            fig, ax = plt.subplots(
                config["net"]["hidden size"], 2, figsize=(20, 20), dpi=dpi
            )
            for i, (name, recordings) in enumerate(neuron_dict.items()):
                for var, vals in recordings.items():
                    if len(vals):
                        for j in range(np.array(vals).shape[1]):
                            ax[j, i].plot(time_list, np.array(vals)[:, j], label=var)
                            ax[j, i].grid(True)

            fig.tight_layout()

            if verbose:
                fig.savefig(f"{config['log location']}neurons+{int(h)}m.png")

        if verbose > 1:
            plt.show()


def vis_disturbance(config, parameters, verbose=2):
    # Build environment
    env = build_environment(config)
    env = randomize_env(env, config)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

    # Reset network and env
    if isinstance(network, SNNNetwork):
        network.reset_state()
    obs = env.reset(h0=config["env"]["h0"][1])
    done = False

    # Indicators whether disturbance already happened
    dist_1 = False
    dist_2 = False

    # For plotting
    state_list = []
    obs_gt_list = []
    obs_list = []
    time_list = []
    encoding_list = []

    # For neuron visualization
    neuron_dict = OrderedDict(
        [
            (name, {"trace": [], "volt": [], "spike": [], "thresh": []})
            for name, child in network.named_children()
            if isinstance(child, BaseNeuron)
        ]
    )

    while not done:
        # Log performance
        state_list.append(env.state.copy())
        obs_gt_list.append(env.div_ph.copy())
        obs_list.append(obs.copy())
        time_list.append(env.t)

        # Log neurons
        for name, child in network.named_children():
            if name in neuron_dict:
                neuron_dict[name]["trace"].append(
                    child.trace.detach().clone().view(-1).numpy()
                )
                neuron_dict[name]["volt"].append(
                    child.v_cell.detach().clone().view(-1).numpy()
                ) if hasattr(child, "v_cell") else None
                neuron_dict[name]["spike"].append(
                    child.spikes.detach().clone().view(-1).numpy()
                ) if hasattr(child, "spikes") else None
                neuron_dict[name]["thresh"].append(
                    child.thresh.detach().clone().view(-1).numpy()
                ) if hasattr(child, "thresh") else None

        # Step the environment
        obs = torch.from_numpy(obs)
        action = network.forward(obs.view(1, 1, -1))
        action = action.numpy()
        if env.t >= 1.5 and not dist_1:
            env.set_disturbance(200.0, 0.0)
            obs, _, done, _ = env.step(action)
            env.unset_disturbance()
            dist_1 = True
        elif env.t >= 2.5 and not dist_2:
            env.set_disturbance(0.0, -2000.0)
            obs, _, done, _ = env.step(action)
            env.unset_disturbance()
            dist_2 = True
        else:
            obs, _, done, _ = env.step(action)

        # Log encoding as well
        if isinstance(network, SNNNetwork):
            encoding_list.append(network.input.view(-1).numpy())

    # Plot
    fig_p, axs_p = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
    # Height
    axs_p[0].plot(time_list, np.array(state_list)[:, 0], label="Height")
    axs_p[0].set_ylabel("height [m]")
    # Velocity
    axs_p[1].plot(time_list, np.array(state_list)[:, 1], label="Velocity")
    axs_p[1].set_ylabel("velocity [m/s]")
    # Acceleration/thrust
    axs_p[2].plot(time_list, np.array(state_list)[:, 2], label="Thrust")
    axs_p[2].set_ylabel("acceleration [m/s2]")
    # Divergence
    axs_p[3].plot(time_list, np.array(obs_gt_list)[:, 0], label="GT divergence")
    axs_p[3].plot(time_list, np.array(obs_list)[:, 0], label="Divergence")
    if isinstance(network, SNNNetwork) and network.encoding == "both":
        axs_p[3].plot(time_list, np.array(encoding_list)[:, 0], label="Encoded +D")
        axs_p[3].plot(time_list, np.array(encoding_list)[:, 2], label="Encoded -D")
    elif isinstance(network, SNNNetwork) and network.encoding == "divergence":
        axs_p[3].plot(time_list, np.array(encoding_list)[:, 0], label="Encoded +D")
        axs_p[3].plot(time_list, np.array(encoding_list)[:, 1], label="Encoded -D")
    axs_p[3].set_ylabel("divergence [1/s]")
    # Divergence dot
    axs_p[4].plot(time_list, np.array(obs_gt_list)[:, 1], label="GT div dot")
    axs_p[4].plot(time_list, np.array(obs_list)[:, 1], label="Div dot")
    if isinstance(network, SNNNetwork) and network.encoding == "both":
        axs_p[4].plot(time_list, np.array(encoding_list)[:, 1], label="Encoded +Ddot")
        axs_p[4].plot(time_list, np.array(encoding_list)[:, 3], label="Encoded -Ddot")
    axs_p[4].set_ylabel("divergence dot [1/s2]")
    axs_p[4].set_xlabel("time [s]")

    for ax in axs_p:
        ax.grid()
        ax.legend()
    plt.tight_layout()

    if verbose:
        plt.savefig(f"{config['log location']}disturbance+performance.png")

    # Plot neurons
    if isinstance(network, SNNNetwork):
        dpi = 50 if config["net"]["hidden size"] > 10 else 100
        fig, ax = plt.subplots(
            config["net"]["hidden size"], 2, figsize=(20, 20), dpi=dpi
        )
        for i, (name, recordings) in enumerate(neuron_dict.items()):
            for var, vals in recordings.items():
                if len(vals):
                    for j in range(np.array(vals).shape[1]):
                        ax[j, i].plot(time_list, np.array(vals)[:, j], label=var)
                        ax[j, i].grid(True)

        fig.tight_layout()

        if verbose:
            fig.savefig(f"{config['log location']}disturbance+neurons.png")

    if verbose > 1:
        plt.show()
