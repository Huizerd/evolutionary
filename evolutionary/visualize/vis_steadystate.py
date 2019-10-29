import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from evolutionary.utils.constructors import build_network
from evolutionary.visualize.colormap import parula_map


def vis_steadystate(config, parameters, verbose=2):
    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

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
            network.reset_state()
            for k in range(time):
                obs = np.array([div[i], divdot[j]])
                obs = torch.from_numpy(obs)
                action = network.forward(obs.view(1, 1, -1))
                response[i, j, k] = action.item()

    # Steady-state check plot
    fig_c, ax_c = plt.subplots()
    ax_c.set_xlabel("steps")
    ax_c.set_ylabel("action")
    ax_c.set_title("corner responses")
    for i in [0, -1]:
        for j in [0, -1]:
            ax_c.plot(
                response[i, j, :], label=f"div: {div_lim[i]}, divdot: {divdot_lim[j]}"
            )

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
    ax_ss[0].set_title("steady state response (non-bounded)")
    ax_ss[0].set_ylabel("divdot [1/s2]")
    ax_ss[0].set_xlabel("div [1/s]")
    cbar = fig_ss.colorbar(im, ax=ax_ss[0])
    cbar.set_label("thrust command [m/s2]")

    # Bounded
    im = ax_ss[1].imshow(
        zi,
        vmin=config["env"]["thrust bounds"][0] * 9.81,
        vmax=config["env"]["thrust bounds"][1] * 9.81,
        cmap=parula_map,
        extent=[*div_lim, *divdot_lim],
        aspect=0.5,
        origin="lower",
    )
    ax_ss[1].set_title("steady state response (bounded)")
    ax_ss[1].set_ylabel("divdot [1/s2]")
    ax_ss[1].set_xlabel("div [1/s]")
    cbar = fig_ss.colorbar(im, ax=ax_ss[1])
    cbar.set_label("thrust command [m/s2]")
    fig_ss.tight_layout()

    if verbose:
        fig_ss.savefig(f"{config['log location']}ss_response.png")

    if verbose > 1:
        plt.show()
