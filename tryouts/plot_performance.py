import argparse
import yaml
import os
import shutil
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8

from pysnn.neuron import BaseNeuron
from pysnn.network import SNNNetwork

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env


def plot_performance(folder, parameters):
    folder = Path(folder)
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder / ("test+" + individual_id)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    # Load config
    with open(folder / "config.yaml", "r") as cf:
        config = yaml.full_load(cf)

    # Build environment
    env = build_environment(config)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))

    # Create plot for performance
    fig_p, axs_p = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
    axs_p[0].set_ylabel("height [m]")
    axs_p[1].set_ylabel("velocity [m/s]")
    axs_p[2].set_ylabel("thrust [m/s2]")
    axs_p[3].set_ylabel("divergence [1/s]")
    axs_p[4].set_ylabel("divergence dot [1/s2]")
    axs_p[5].set_ylabel("spikes [?]")
    axs_p[5].set_xlabel("time [s]")

    # Create plot for neurons
    fig_n, axs_n = plt.subplots(7, 3, sharex=True, figsize=(10, 10))
    axs_n = axs_n.flatten()

    # Create list to hold spike rates per neuron
    rates = []

    # 5 runs
    for i in range(5):
        # With different properties
        # Randomizing here means that another run of this file will get different envs,
        # but so be it. Not easy to change
        env = randomize_env(env, config)
        # Reset network and env
        if isinstance(network, SNNNetwork):
            network.reset_state()
        obs = env.reset(h0=config["env"]["h0"][1])
        done = False
        spikes = 0

        # For plotting
        state_list = []
        obs_gt_list = []
        obs_list = []
        time_list = []
        spike_list = []

        # For neuron visualization
        neuron_dict = OrderedDict(
            [
                (name, {"trace": [], "spike": []})
                for name, child in network.named_children()
                if isinstance(child, BaseNeuron)
            ]
        )

        # Log performance
        state_list.append(env.state.copy())
        obs_gt_list.append(env.div_ph.copy())
        obs_list.append(obs.copy())
        time_list.append(env.t)
        spike_list.append([0, 0])
        # Log neurons
        for name, child in network.named_children():
            if name in neuron_dict:
                neuron_dict[name]["trace"].append(
                    child.trace.detach().clone().view(-1).numpy()
                )
                neuron_dict[name]["spike"].append(
                    child.spikes.detach().clone().view(-1).numpy()
                ) if hasattr(child, "spikes") else None

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)

            # Log performance
            state_list.append(env.state.copy())
            obs_gt_list.append(env.div_ph.copy())
            obs_list.append(obs.copy())
            time_list.append(env.t)
            if isinstance(network, SNNNetwork):
                spikes += (
                    network.neuron1.spikes.sum().item()
                    + network.neuron2.spikes.sum().item()
                )
                spike_list.append(
                    [
                        spikes,
                        network.neuron1.spikes.sum().item()
                        + network.neuron2.spikes.sum().item(),
                    ]
                )

            # Log neurons
            for name, child in network.named_children():
                if name in neuron_dict:
                    neuron_dict[name]["trace"].append(
                        child.trace.detach().clone().view(-1).numpy()
                    )
                    neuron_dict[name]["spike"].append(
                        child.spikes.detach().clone().view(-1).numpy()
                    ) if hasattr(child, "spikes") else None

        # Plot data
        # Height
        axs_p[0].plot(time_list, np.array(state_list)[:, 0], "C0", label=f"run {i}")
        # Velocity
        axs_p[1].plot(time_list, np.array(state_list)[:, 1], "C0", label=f"run {i}")
        # Thrust
        axs_p[2].plot(time_list, np.array(state_list)[:, 2], "C0", label=f"run {i}")
        # Divergence
        axs_p[3].plot(time_list, np.array(obs_list)[:, 0], "C0", label=f"run {i}")
        axs_p[3].plot(time_list, np.array(obs_gt_list)[:, 0], "C1", label=f"run {i} GT")
        # Divergence dot
        axs_p[4].plot(time_list, np.array(obs_list)[:, 1], "C0", label=f"run {i}")
        axs_p[4].plot(time_list, np.array(obs_gt_list)[:, 1], "C1", label=f"run {i} GT")
        # Spikes
        axs_p[5].plot(
            time_list,
            np.array(spike_list)[:, 0] / np.array(time_list),
            "C0",
            label=f"run {i}",
        )
        axs_p[5].plot(
            time_list,
            pd.Series(np.array(spike_list)[:, 1])
            .rolling(window=20, min_periods=1)
            .mean()
            .values,
            "C1",
            label=f"run {i}",
        )

        # Plot neurons
        neurons = OrderedDict()
        k = 0
        # Go over layers
        for recordings in neuron_dict.values():
            # Go over neurons in layer
            for j in range(np.array(recordings["spike"]).shape[1]):
                neurons[f"n{k}_spike"] = np.array(recordings["spike"])[:, j].astype(
                    float
                )
                neurons[f"n{k}_trace"] = np.array(recordings["trace"])[:, j]
                neurons[f"n{k}_ma"] = (
                    pd.Series(np.array(recordings["spike"])[:, j])
                    .rolling(window=20, min_periods=1)
                    .mean()
                    .values
                )
                axs_n[k].plot(time_list, np.array(recordings["trace"])[:, j], "C0")
                axs_n[k].plot(time_list, np.array(recordings["spike"])[:, j], "C1")
                axs_n[k].plot(
                    time_list,
                    pd.Series(np.array(recordings["spike"])[:, j])
                    .rolling(window=20, min_periods=1)
                    .mean()
                    .values,
                    "C2",
                )
                axs_n[k].set_title(f"{k}")
                k += 1

        # Save run
        rates.append(
            [
                [
                    v.sum() / (time_list[-1] - config["env"]["settle"]),
                    v.sum() / (len(time_list) - config["env"]["settle"] // env.dt + 1),
                ]
                for k, v in neurons.items()
                if "spike" in k
            ]
        )
        data = pd.DataFrame(
            {
                "time": time_list,
                "pos_z": np.array(state_list)[:, 0],
                "vel_z": np.array(state_list)[:, 1],
                "thrust": np.array(state_list)[:, 2],
                "div": np.array(obs_list)[:, 0],
                "div_gt": np.array(obs_gt_list)[:, 0],
                "divdot": np.array(obs_list)[:, 1],
                "divdot_gt": np.array(obs_gt_list)[:, 1],
                "spike_count": np.array(spike_list)[:, 0],
                "spike_step": np.array(spike_list)[:, 1],
            }
        )
        neurons = pd.DataFrame(neurons)
        data = pd.concat([data, neurons], 1)
        data.to_csv(str(save_folder) + f"/run{i}.csv", index=False, sep=",")

    # Compute rates
    rates = pd.DataFrame(
        {
            "mean_time": np.array(rates).mean(0)[:, 0],
            "mean_steps": np.array(rates).mean(0)[:, 1],
            "std_time": np.array(rates).std(0)[:, 0],
            "std_steps": np.array(rates).std(0)[:, 1],
        }
    )
    rates.to_csv(str(save_folder) + f"/rates.csv", index=False, sep=",")

    # Write network to tikz-network-compatible file
    # Edges
    # First layer
    edges_0 = pd.DataFrame(columns=["u", "v", "lw_raw", "color", "lw"])
    for i in range(network.fc1.weight.shape[0]):
        for j in range(-network.fc1.weight.shape[1], 0):
            edges_0 = edges_0.append({"u": j, "v": i, "lw": 0.0}, ignore_index=True)
    edges_0["u"] = edges_0["u"].astype(int)
    edges_0["v"] = edges_0["v"].astype(int)
    edges_0["lw_raw"] = network.fc1.weight.view(-1).tolist()
    edges_0["color"] = np.where(edges_0["lw_raw"] > 0, "magenta", "cyan")
    edges_0["lw"] = edges_0["lw_raw"].abs()
    # Second layer
    edges_1 = pd.DataFrame(columns=["u", "v", "lw_raw", "color", "lw"])
    for i in range(network.fc2.weight.shape[0]):
        for j in range(network.fc2.weight.shape[1]):
            edges_1 = edges_1.append(
                {"u": j, "v": i + network.fc2.weight.shape[1], "lw": 0.0},
                ignore_index=True,
            )
    edges_1["u"] = edges_1["u"].astype(int)
    edges_1["v"] = edges_1["v"].astype(int)
    edges_1["lw_raw"] = network.fc2.weight.view(-1).tolist()
    edges_1["color"] = np.where(edges_1["lw_raw"] > 0, "magenta", "cyan")
    edges_1["lw"] = edges_1["lw_raw"].abs()
    edges = pd.concat([edges_0, edges_1], 0)
    edges.to_csv(str(save_folder) + f"/network_edges.csv", index=False, sep=",")

    # Vertices
    k = 0
    # Input layer
    vertices_0 = pd.DataFrame(columns=["id", "x", "y", "color"])
    for j in range(-network.fc1.weight.shape[1], 0):
        vertices_0 = vertices_0.append(
            {
                "id": j,
                "x": 0.0,
                "y": network.fc1.weight.shape[1] / 4
                - 0.25
                - 0.5 * (network.fc1.weight.shape[1] + j),
                "color": "cyan",
            },
            ignore_index=True,
        )
    # Hidden layer
    vertices_1 = pd.DataFrame(columns=["id", "x", "y", "color"])
    for j in range(network.fc2.weight.shape[1]):
        vertices_1 = vertices_1.append(
            {
                "id": j,
                "x": 2.0,
                "y": network.fc2.weight.shape[1] / 4 - 0.25 - 0.5 * j,
                "color": f"cyan!{100 - 3.333 * rates['mean_time'][k]}!magenta",
            },
            ignore_index=True,
        )
        k += 1
    # Output layer
    vertices_2 = pd.DataFrame(columns=["id", "x", "y", "color"])
    vertices_2 = vertices_2.append(
        {
            "id": network.fc2.weight.shape[1],
            "x": 4.0,
            "y": 0.0,
            "color": f"cyan!{100 - 3.333 * rates['mean_time'][k]}!magenta",
        },
        ignore_index=True,
    )
    vertices = pd.concat([vertices_0, vertices_1, vertices_2], 0)
    vertices.to_csv(str(save_folder) + f"/network_vertices.csv", index=False, sep=",")

    # Add grid
    for ax in axs_p:
        ax.grid()
    fig_p.tight_layout()
    fig_n.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_performance(args["folder"], args["parameters"])
