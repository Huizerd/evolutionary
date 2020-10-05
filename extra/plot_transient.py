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
    time_list = []
    action_list = []
    obs_list = []
    obs_error_list = []
    input_spike_list = []
    output_spike_list = []
    output_trace_list = []

    for i in range(100):
        env = randomize_env(env, config)
        network.reset_state()
        obs = env.reset(h0=(config["env"]["h0"][-1] + config["env"]["h0"][0]) / 2)
        done = False

        # For plotting
        time = []
        actions = []
        observations = []
        obs_errors = []
        in_spikes = []
        out_spikes = []
        out_trace = []

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = network.forward(obs.clone().view(1, 1, -1))
            action = action.numpy()

            if env.t >= env.settle:
                time.append(env.t)
                actions.append(np.clip(env.action[0], *config["env"]["g bounds"]))
                observations.append(obs.numpy().copy())
                obs_errors.append(
                    obs.numpy().copy() - np.array([config["evo"]["D setpoint"], 0.0])
                )
                in_spikes.append(network.input.view(-1).numpy().copy())
                out_spikes.append(network.out_spikes.float().numpy().copy())
                out_trace.append(network.out_trace.numpy().copy())

            obs, _, done, _ = env.step(action)

        time_list.append(time)
        action_list.append(actions)
        obs_list.append(observations)
        obs_error_list.append(obs_errors)
        input_spike_list.append(in_spikes)
        output_spike_list.append(out_spikes)
        output_trace_list.append(out_trace)

    # Visualize
    all_x = []
    all_y = []
    fig, ax = plt.subplots(1, 1)
    for i, tim, act, ob, ober, ins, outs, outt in zip(
        range(len(action_list)),
        time_list,
        action_list,
        obs_list,
        obs_error_list,
        input_spike_list,
        output_spike_list,
        output_trace_list,
    ):
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
        output = pd.DataFrame(
            {"time": np.array(tim), "x": np.array(ob)[sort_idx, 0], "y": ma}
        )
        output_raw = pd.DataFrame(
            {
                "time": np.array(tim),
                "D": np.array(ob)[:, 0],
                "Derror": np.array(ober)[:, 0],
                "Tsp": act,
            }
        )
        input_spikes_raw = pd.DataFrame(
            np.concatenate((np.array(tim)[..., None], np.array(ins)), axis=1),
            columns=["time"]
            + [
                f"Derror={val1:.2f}|{val2:.2f}"
                for val1, val2 in zip(
                    [-99.0, *network.buckets.tolist()],
                    [*network.buckets.tolist(), 99.0],
                )
            ],
        )
        output_spikes_raw = pd.DataFrame(
            np.concatenate((np.array(tim)[..., None], np.array(outs)), axis=1),
            columns=["time"] + [f"Tsp={val:.2f}" for val in network.trace_weights],
        )
        output_traces_raw = pd.DataFrame(
            np.concatenate((np.array(tim)[..., None], np.array(outt)), axis=1),
            columns=["time"] + [f"Tsp={val:.2f}" for val in network.trace_weights],
        )

        output.to_csv(save_folder + f"run{i}.csv", index=False, sep=",")
        output_raw.to_csv(save_folder + f"run_raw{i}.csv", index=False, sep=",")
        input_spikes_raw.to_csv(
            save_folder + f"run_raw_in_spikes{i}.csv", index=False, sep=","
        )
        output_spikes_raw.to_csv(
            save_folder + f"run_raw_out_spikes{i}.csv", index=False, sep=","
        )
        output_traces_raw.to_csv(
            save_folder + f"run_raw_out_traces{i}.csv", index=False, sep=","
        )

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
