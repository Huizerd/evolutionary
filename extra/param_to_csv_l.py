import argparse
import os
import yaml

import torch
import pandas as pd
import numpy as np

from pysnn.neuron import AdaptiveLIFNeuron

from evolutionary.network.snn import TwoLayerSNN, ThreeLayerSNN
from evolutionary.utils.constructors import build_network


def param_to_csv_l(folder, parameters):
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder + f"/param_l+{individual_id}/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load config
    with open(folder + "/config.yaml", "r") as f:
        config = yaml.full_load(f)

    # Build network
    network = build_network(config)
    # Load network parameters
    network.load_state_dict(torch.load(parameters))

    # Two-layer
    if isinstance(network, TwoLayerSNN):
        # Hidden neuron
        if isinstance(network.neuron1, AdaptiveLIFNeuron):
            raise NotImplementedError("AdaptiveLIFNeuron is not yet supported")
        else:
            hid = pd.DataFrame(
                np.concatenate(
                    [
                        (network.neuron1.tau_v.view(-1, 1).data.numpy() * 4096).round(),
                        (network.neuron1.thresh.view(-1, 1).data.numpy() * 128).round(),
                    ],
                    axis=1,
                ),
                columns=["tau_v", "thresh"],
            )
            hid.to_csv(save_folder + "hid.csv")

        # Output neuron
        if isinstance(network.neuron2, AdaptiveLIFNeuron):
            raise NotImplementedError("AdaptiveLIFNeuron is not yet supported")
        else:
            out = pd.DataFrame(
                np.concatenate(
                    [
                        network.neuron2.alpha_t.view(-1, 1).data.numpy(),
                        (network.neuron2.tau_v.view(-1, 1).data.numpy() * 4096).round(),
                        network.neuron2.tau_t.view(-1, 1).data.numpy(),
                        (network.neuron2.thresh.view(-1, 1).data.numpy() * 128).round(),
                    ],
                    axis=1,
                ),
                columns=["alpha_t", "tau_v", "tau_t", "thresh"],
            )
            out.to_csv(save_folder + "out.csv")

    # Three-layer
    elif isinstance(network, ThreeLayerSNN):
        raise NotImplementedError(
            "Parameter export not yet implemented for ThreeLayerSNN"
        )


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    param_to_csv_l(args["folder"], args["parameters"])
