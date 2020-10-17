import argparse
import yaml

import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["lines.linewidth"] = 0.8


from evolutionary.utils.constructors import build_network


def compare_float_int(folder, parameters, run):

    # Load config
    with open(folder + "/config.yaml", "r") as f:
        config = yaml.full_load(f)

    # Load network
    network = build_network(config)
    network.load_state_dict(torch.load(parameters))
    network.reset_state()

    # Load data
    data = pd.read_csv(run, sep=",")

    # Print warning
    print()
    print(
        "The supplied run must be with an integer tau_v, while now you must have a float tau_v"
    )
    print("Don't forget to adapt CustomLIFNeuron to do nothing!")
    print()

    # Convert tau_v
    network.neuron1.tau_v = (4096 - network.neuron1.tau_v) / 4096
    network.neuron2.tau_v = (4096 - network.neuron2.tau_v) / 4096

    time = []
    divergence = []
    tsp = []
    for i in range(data.shape[0]):
        obs = torch.as_tensor(data["div"][i]).view(1, 1, -1)
        action = network.forward(obs)
        divergence.append(data["div"][i])
        tsp.append(action.item())
        time.append(data["time"][i])

    plt.plot(data["time"], data["tsp"])
    plt.plot(time, tsp)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    parser.add_argument("--run", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    compare_float_int(args["folder"], args["parameters"], args["run"])
