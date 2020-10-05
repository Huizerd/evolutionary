import argparse
import yaml
import os
import shutil
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8

from pysnn.neuron import BaseNeuron

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env


def plot_pid(folder):
    save_folder = folder + f"/test"
    suffix = 0
    while os.path.exists(f"{save_folder}+{str(suffix)}/"):
        suffix += 1
    save_folder += f"+{str(suffix)}/"
    os.makedirs(save_folder)

    # Load config
    with open(folder + "/config.yaml", "r") as f:
        config = yaml.full_load(f)

    # Controller config
    p_gain = 0.98
    d_setpoint = 0.5
    tmin = -0.2
    tmax = 0.25

    # Build environment
    env = build_environment(config)

    # Create plot for performance
    fig_p, axs_p = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
    axs_p[0].set_ylabel("height [m]")
    axs_p[1].set_ylabel("velocity [m/s]")
    axs_p[2].set_ylabel("thrust setpoint [g]")
    axs_p[3].set_ylabel("divergence [1/s]")
    axs_p[4].set_ylabel("divergence dot [1/s2]")
    axs_p[5].set_ylabel("spikes [?]")
    axs_p[5].set_xlabel("time [s]")

    # 5 runs
    for i in range(5):
        # With different properties
        # Randomizing here means that another run of this file will get different envs,
        # but so be it. Not easy to change
        env = randomize_env(env, config)
        obs = env.reset(h0=(config["env"]["h0"][-1] + config["env"]["h0"][0]) / 2)
        env.action = np.array(
            [env.action]
        )  # obscure fix to allow logging after env step
        done = False

        # For plotting
        action_list = []
        state_list = []
        obs_gt_list = []
        obs_list = []
        time_list = []

        # Log performance
        action_list.append(np.clip(env.action[0], *config["env"]["g bounds"]))
        state_list.append(env.state.copy())
        obs_gt_list.append(env.div_ph.copy())
        obs_list.append(obs.copy())
        time_list.append(env.t)
        while not done:
            # Step the environment
            d_error = obs[0] - d_setpoint
            action = max(tmin, min(tmax, p_gain * d_error))
            action = np.array(action)[None]

            # Log performance
            action_list.append(np.clip(env.action[0], *config["env"]["g bounds"]))
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())
            time_list.append(env.t)

            # Advance env
            obs, _, done, _ = env.step(action)

        # Plot data
        # Height
        axs_p[0].plot(time_list, np.array(state_list)[:, 0], "C0", label=f"run {i}")
        # Velocity
        axs_p[1].plot(time_list, np.array(state_list)[:, 1], "C0", label=f"run {i}")
        # Thrust
        axs_p[2].plot(time_list, np.array(action_list), "C0", label=f"run {i}")
        # Divergence
        axs_p[3].plot(time_list, np.array(obs_list)[:, 0], "C0", label=f"run {i}")
        axs_p[3].plot(time_list, np.array(obs_gt_list)[:, 0], "C1", label=f"run {i} GT")
        # Divergence dot
        axs_p[4].plot(time_list, np.array(obs_list)[:, 1], "C0", label=f"run {i}")
        axs_p[4].plot(time_list, np.array(obs_gt_list)[:, 1], "C1", label=f"run {i} GT")

        # Save run
        data = pd.DataFrame(
            {
                "time": time_list,
                "pos_z": np.array(state_list)[:, 0],
                "vel_z": np.array(state_list)[:, 1],
                "thrust": np.array(state_list)[:, 2],
                "tsp": np.array(action_list),
                "tsp_lp": pd.Series(action_list)
                .rolling(window=20, min_periods=1)
                .mean()
                .values,
                "div": np.array(obs_list)[:, 0],
                "div_gt": np.array(obs_gt_list)[:, 0],
                "divdot": np.array(obs_list)[:, 1],
                "divdot_gt": np.array(obs_gt_list)[:, 1],
                "spike_count": np.zeros(len(time_list)),
            }
        )
        data.to_csv(save_folder + f"run{i}.csv", index=False, sep=",")

    # Add grid
    for ax in axs_p:
        ax.grid()
    fig_p.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_pid(args["folder"])
