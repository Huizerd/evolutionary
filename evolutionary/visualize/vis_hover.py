import torch
import numpy as np
import matplotlib.pyplot as plt

from gym_quad.envs import QuadHover

from evolutionary.network.ann import ANN


def vis_hover(weights):
    # Build environment with wind, test for various starting altitudes
    env = QuadHover(
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
    h0 = [2.0, 4.0, 6.0, 8.0]

    # Load network
    network = ANN(2, 8, 1).load_state_dict(torch.load(weights))

    # For plotting
    state_list = []
    obs_gt_list = []
    obs_list = []

    for h in h0:
        obs = env.reset(h0=h)
        done = False

        while not done:
            # Log everything
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())

            obs = torch.from_numpy(obs)
            action = network.forward(obs)
            action = action.numpy()
            obs, _, done, _ = env.step(action)

        # Plot
        plt.plot(np.array(state_list)[:, 0], label="Height")
        plt.plot(np.array(state_list)[:, 1], label="Velocity")
        plt.plot(np.array(state_list)[:, 2], label="Thrust")
        plt.plot(np.array(obs_gt_list)[:, 0], label="GT divergence")
        plt.plot(np.array(obs_gt_list)[:, 1], label="GT div dot")
        plt.plot(np.array(obs_list)[:, 0], label="Divergence")
        plt.plot(np.array(obs_list)[:, 1], label="Div dot")
        plt.xlabel("Time")
        plt.title(f"Performance starting from {h} m")
        plt.legend()
        plt.grid()
        plt.show()
