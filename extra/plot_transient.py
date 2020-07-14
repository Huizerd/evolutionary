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

from evolutionary.utils.constructors import build_network, build_environment
from evolutionary.utils.utils import randomize_env


def plot_transient(folder, parameters):
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder + f"/transient+{individual_id}"
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
    action_list = []
    obs_list = []
    for i in range(100):
        env = randomize_env(env, config)
        network.reset_state()
        obs = env.reset(h0=config["env"]["h0"][1])
        done = False

        # For plotting
        actions = []
        observations = []

        actions.append(np.clip(env.action, *config["env"]["g bounds"]))
        observations.append(obs.copy())

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)

            if env.t >= env.settle:
                actions.append(np.clip(env.action[0], *config["env"]["g bounds"]))
                observations.append(obs.copy())

        action_list.append(actions[1:])
        obs_list.append(observations[1:])

    # Visualize
    all_x = []
    all_y = []
    fig, ax = plt.subplots(1, 1)
    for i, act, ob in zip(range(len(action_list)), action_list, obs_list):
        sort_idx = np.argsort(np.array(ob)[:, 0])
        ma = (
            pd.Series(np.array(act)[sort_idx])
            .rolling(window=40, min_periods=1)
            .mean()
            .values
        )
        ax.plot(np.array(ob)[sort_idx, 0], ma, "r", alpha=0.5)
        all_x.extend((np.array(ob)[:, 0]).tolist())
        all_y.extend(act)
        output = pd.DataFrame({"x": np.array(ob)[sort_idx, 0], "y": ma})
        output.to_csv(save_folder + f"run{i}.csv", index=False, sep=",")

    ax.scatter(all_x, all_y, c="b", alpha=0.5)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-0.9, 0.9])
    scatter = pd.DataFrame({"x": all_x, "y": all_y})
    scatter.to_csv(save_folder + f"raw_points.csv", index=False, sep=",")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    plot_transient(args["folder"], args["parameters"])
