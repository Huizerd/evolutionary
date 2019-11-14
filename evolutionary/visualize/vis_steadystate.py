import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from pysnn.network import SNNNetwork

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env
from evolutionary.visualize.colormap import parula_map


def vis_steadystate(config, parameters, verbose=2):
    # Build environment
    env = build_environment(config)
    env = randomize_env(env, config)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

    # Do one run from 5m
    if isinstance(network, SNNNetwork):
        network.reset_state()
    obs = env.reset(h0=(config["env"]["h0"][0] + config["env"]["h0"][-1]) / 2)
    done = False

    # For plotting
    obs_gt_list = []

    while not done:
        # Log performance
        obs_gt_list.append(env.div_ph.copy())

        # Step the environment
        obs = torch.from_numpy(obs)
        action = network.forward(obs.view(1, 1, -1))
        action = action.numpy()
        obs, _, done, _ = env.step(action)

    # Convert to array
    obs_gt = np.array(obs_gt_list)

    # Input: D in [-10, 10], Ddot in [-100, 100]
    div_lim = [-10.0, 10.0]
    divdot_lim = [-20.0, 20.0]
    div = np.linspace(*div_lim, 101)
    divdot = np.linspace(*divdot_lim, 101)

    # Batch size is 1, so no parallel stuff
    # Show each input for 100 steps
    time = 100
    response = np.zeros((div.shape[0], divdot.shape[0], time))

    # Start loop
    for i in range(div.shape[0]):
        for j in range(divdot.shape[0]):
            if isinstance(network, SNNNetwork):
                network.reset_state()
            for k in range(time):
                obs = np.array([div[i], divdot[j]])
                obs = torch.from_numpy(obs).float()
                action = network.forward(obs.view(1, 1, -1))
                response[i, j, k] = action.item() * config["env"]["g"]

    # Steady-state corner plot
    colors = ["xkcd:neon red", "xkcd:neon blue", "xkcd:neon green", "xkcd:neon purple"]
    c = 0
    fig_c, ax_c = plt.subplots()
    ax_c.set_xlabel("steps")
    ax_c.set_ylabel("action")
    ax_c.set_title("corner responses")
    for d in div_lim:
        for dd in divdot_lim:
            if isinstance(network, SNNNetwork):
                network.reset_state()
            corner = []
            corner_noise = []
            for _ in range(time):
                obs = np.array([d, dd])
                obs_noise = (
                    obs
                    + np.random.normal(0.0, env.noise_std)
                    + abs(obs) * np.random.normal(0.0, env.noise_p_std)
                )
                obs = torch.from_numpy(obs).float()
                obs_noise = torch.from_numpy(obs_noise).float()
                action = network.forward(obs.view(1, 1, -1))
                action_noise = network.forward(obs_noise.view(1, 1, -1))
                corner.append(action.item() * config["env"]["g"])
                corner_noise.append(action_noise.item() * config["env"]["g"])
            ax_c.plot(corner, label=f"div: {d}, divdot: {dd}", color=colors[c])
            ax_c.plot(corner_noise, color=colors[c])
            c += 1

    ax_c.legend()
    ax_c.grid()

    if verbose:
        fig_c.savefig(f"{config['log location']}ss_corners.png")

    # Interpolate
    x = np.linspace(*div_lim, 2100)
    y = np.linspace(*divdot_lim, 2100)
    xi, yi = np.meshgrid(x, y)
    response = response[:, :, -50:].mean(-1)
    interp = RectBivariateSpline(div, divdot, response)
    zi = interp.ev(xi, yi)

    # Visualize
    fig_ss, ax_ss = plt.subplots(1, 2, figsize=(10, 5))

    # Non-bounded
    im = ax_ss[0].imshow(
        zi,
        vmin=response.min(),
        vmax=response.max(),
        cmap=parula_map,
        extent=[*div_lim, *divdot_lim],
        aspect=0.5,
        origin="lower",
    )
    ax_ss[0].plot(obs_gt[:, 0], obs_gt[:, 1], "r")
    ax_ss[0].set_title("steady state response (non-bounded)")
    ax_ss[0].set_ylabel("divergence dot [1/s2]")
    ax_ss[0].set_xlabel("divergence [1/s]")
    ax_ss[0].set_xlim(div_lim)
    ax_ss[0].set_ylim(divdot_lim)
    cbar = fig_ss.colorbar(im, ax=ax_ss[0])
    cbar.set_label("thrust command [m/s2]")

    # Bounded
    im = ax_ss[1].imshow(
        zi,
        vmin=config["env"]["g bounds"][0] * config["env"]["g"],
        vmax=config["env"]["g bounds"][1] * config["env"]["g"],
        cmap=parula_map,
        extent=[*div_lim, *divdot_lim],
        aspect=0.5,
        origin="lower",
    )
    ax_ss[1].plot(obs_gt[:, 0], obs_gt[:, 1], "r")
    ax_ss[1].set_title("steady state response (bounded)")
    ax_ss[1].set_ylabel("divergence dot [1/s2]")
    ax_ss[1].set_xlabel("divergence [1/s]")
    ax_ss[1].set_xlim(div_lim)
    ax_ss[1].set_ylim(divdot_lim)
    cbar = fig_ss.colorbar(im, ax=ax_ss[1])
    cbar.set_label("thrust command [m/s2]")
    fig_ss.tight_layout()

    if verbose:
        fig_ss.savefig(f"{config['log location']}ss_response.png")

    if verbose > 1:
        plt.show()
