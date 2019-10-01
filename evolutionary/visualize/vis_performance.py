from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt

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
        thrust_tc=0.02,
        settle=1.0,
        wind=0.1,
        h0=5.0,
        dt=0.02,
        seed=0,
    )
    h0 = config["env"]["h0"]

    # Load network
    if config["network"] == "ANN":
        network = ANN(2, config["hidden size"], 1)
    elif config["network"] == "SNN":
        network = SNN(2, config["hidden size"], 1)
    else:
        raise KeyError("Not a valid network key!")

    network.load_state_dict(torch.load(parameters))

    for h in h0:
        obs = env.reset(h0=h)
        done = False

        # For plotting
        state_list = []
        obs_gt_list = []
        obs_list = []
        time_list = []

        while not done:
            # Log everything
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())
            time_list.append(env.t)

            obs = torch.from_numpy(obs)
            action = network.forward(obs)
            action = action.numpy()
            obs, _, done, _ = env.step(action)

        # Plot
        plt.plot(time_list, np.array(state_list)[:, 0], label="Height")
        plt.plot(time_list, np.array(state_list)[:, 1], label="Velocity")
        plt.plot(time_list, np.array(state_list)[:, 2], label="Thrust")
        plt.plot(time_list, np.array(obs_gt_list)[:, 0], label="GT divergence")
        plt.plot(time_list, np.array(obs_gt_list)[:, 1], label="GT div dot")
        plt.plot(time_list, np.array(obs_list)[:, 0], label="Divergence")
        # plt.plot(time_list, np.array(obs_list)[:, 1], label="Div dot")
        plt.xlabel("Time")
        plt.title(f"Performance starting from {h} m")
        plt.ylim(-1, env.MAX_H + 1)
        plt.legend()
        plt.grid()
        plt.show()
