import argparse
import yaml
import os
import shutil

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8

from evolutionary.network.snn import TwoLayerSNN
from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env


def plot_transient_neurons(folder, parameters):
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder + f"/transient+neurons+{individual_id}"
    suffix = 0
    while os.path.exists(f"{save_folder}+{str(suffix)}/"):
        suffix += 1
    save_folder += f"+{str(suffix)}/"
    os.makedirs(save_folder)

    # Load config
    with open(folder + "/config.yaml", "r") as cf:
        config = yaml.full_load(cf)

    # Build environment
    env = build_environment(config)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))
    network.reset_state()

    # 100 runs
    time_list = []
    obs_list = []
    obs_error_list = []
    neuron_spike_list = []

    for i in range(10):
        env = randomize_env(env, config)
        network.reset_state()
        obs = env.reset(h0=(config["env"]["h0"][-1] + config["env"]["h0"][0]) / 2)
        env.action = np.array(
            [env.action]
        )  # obscure fix to allow logging after env step
        done = False

        # For plotting
        time = []
        observations = []
        obs_errors = []
        neuron_spikes = []

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.clone().view(1, 1, -1))
            action = action.numpy()

            time.append(env.t)
            observations.append(obs.numpy().copy())
            obs_errors.append(
                obs.numpy().copy() - np.array([config["evo"]["D setpoint"], 0.0])
            )
            if isinstance(network, TwoLayerSNN):
                in_spikes = network.input.view(-1).numpy().copy()
                hid_spikes = network.neuron1.spikes.float().view(-1).numpy().copy()
                out_spikes = network.out_spikes.float().view(-1).numpy().copy()
                neuron_spikes.append(
                    np.concatenate((in_spikes, hid_spikes, out_spikes))
                )
            else:
                raise ValueError(f"Incompatible network type specified")

            obs, _, done, _ = env.step(action)

        time_list.append(time)
        obs_list.append(observations)
        obs_error_list.append(obs_errors)
        neuron_spike_list.append(neuron_spikes)

    # Activity vs D error
    # Activity vs time
    fig1, ax1 = plt.subplots(
        max(config["net"]["layer sizes"]),
        len(config["net"]["layer sizes"]),
        sharex=True,
        sharey=True,
        figsize=(12, 18),
    )
    fig2, ax2 = plt.subplots(
        max(config["net"]["layer sizes"]),
        len(config["net"]["layer sizes"]),
        sharex=True,
        sharey=True,
        figsize=(12, 18),
    )
    n_plots_oe = [[] for _ in range(sum(config["net"]["layer sizes"]))]
    n_plots_act = [[] for _ in range(sum(config["net"]["layer sizes"]))]

    for tim, ob, ober, neur in zip(
        time_list, obs_list, obs_error_list, neuron_spike_list
    ):
        for t, o, oe, n in zip(tim, ob, ober, neur):
            for i in range(len(n_plots_oe)):
                n_plots_oe[i].append(oe[0])
                n_plots_act[i].append(n[i])

    k = 0
    for i in range(len(config["net"]["layer sizes"])):
        for j in range(config["net"]["layer sizes"][i]):
            sort_idx = np.argsort(n_plots_oe[k])
            ma = (
                pd.Series(np.array(n_plots_act[k])[sort_idx])
                .rolling(window=40, min_periods=1)
                .mean()
                .values
            )
            ax1[j, i].plot(np.array(n_plots_oe[k])[sort_idx], ma)
            k += 1

    k = 0
    for i in range(len(config["net"]["layer sizes"])):
        for j in range(config["net"]["layer sizes"][i]):
            for tim, neur in zip(time_list, neuron_spike_list):
                ma = (
                    pd.Series(np.array(neur)[:, k])
                    .rolling(window=40, min_periods=1)
                    .mean()
                    .values
                )
                ax2[j, i].plot(tim, ma)
            k += 1

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_transient_neurons(args["folder"], args["parameters"])
