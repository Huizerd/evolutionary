from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt

from pysnn.neuron import Input, Neuron

from gym_quad.envs import QuadHover as QuadBase
from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN


def vis_performance(config, parameters):
    # Build environment with wind, test for various starting altitudes
    # Use regular base QuadHover (no need for modified reward here)
    env = QuadBase(
        delay=3,
        comp_delay_prob=0.0,
        noise=0.1,
        noise_p=0.1,
        thrust_bounds=(-0.8, 0.5),
        thrust_tc=0.02,
        settle=1.0,
        wind=0.0,
        h0=5.0,
        dt=0.02,
        seed=0,
    )
    h0 = config["env"]["h0"]

    # Load network
    if config["network"] == "ANN":
        network = ANN(2, config["hidden size"], 1)
        vis_neurons = True
    elif config["network"] == "SNN":
        network = SNN(2, config["hidden size"], 1, config)
        vis_neurons = True
    else:
        raise KeyError("Not a valid network key!")

    network.load_state_dict(torch.load(parameters))

    # Go over all heights we trained for
    for h in h0:
        obs = env.reset(h0=h)
        done = False

        # For plotting
        state_list = []
        obs_gt_list = []
        obs_list = []
        time_list = []

        # For neuron visualization
        neuron_dict = OrderedDict(
            [
                (name, {"trace": [], "volt": [], "spike": [], "thresh": []})
                for name, module in network.named_modules()
                if isinstance(module, Input) or isinstance(module, Neuron)
            ]
        )

        while not done:
            # Log performance
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())
            time_list.append(env.t)

            # Log neurons
            # TODO: spike state is being reset before logging, so no use
            for name, module in network.named_modules():
                if name in neuron_dict:
                    neuron_dict[name]["trace"].append(
                        module.trace.T.view(-1).clone().numpy()
                    )
                    neuron_dict[name]["volt"].append(
                        module.v_cell.T.view(-1).clone().numpy()
                    ) if hasattr(module, "v_cell") else None
                    # neuron_dict[name]["spike"].append(module.spiking().view(-1).clone().numpy()) if hasattr(module, "spiking") else None
                    neuron_dict[name]["thresh"].append(
                        module.thresh.T.view(-1).clone().numpy()
                    ) if hasattr(module, "thresh") else None

            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)

        # Plot
        plt.plot(time_list, -np.array(state_list)[:, 2], label="Thrust")
        plt.plot(time_list, np.array(state_list)[:, 0], label="Height")
        plt.plot(time_list, np.array(state_list)[:, 1], label="Velocity")
        plt.plot(time_list, np.array(obs_gt_list)[:, 0], label="GT divergence")
        # plt.plot(time_list, np.array(obs_gt_list)[:, 1], label="GT div dot")
        plt.plot(time_list, np.array(obs_list)[:, 0], label="Divergence")
        # plt.plot(time_list, np.array(obs_list)[:, 1], label="Div dot")
        plt.xlabel("Time")
        plt.title(f"Performance starting from {h} m")
        plt.ylim(-1, env.MAX_H + 1)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Plot neurons
        if vis_neurons:
            fig, ax = plt.subplots(config["hidden size"], 3, figsize=(10, 10))
            for i, (name, recordings) in enumerate(neuron_dict.items()):
                for var, vals in recordings.items():
                    if len(vals):
                        for j in range(np.array(vals).shape[1]):
                            ax[j, i].plot(time_list, np.array(vals)[:, j], label=var)
                            ax[j, i].grid(True)
                            # ax[j, i].set_title(f"{name}: {j}")
                            # ax[j, i].set_xlabel("Time")
                            # ax[j, i].legend()

            fig.tight_layout()

        plt.show()
