import argparse
import os
import yaml

import torch

from pysnn.neuron import AdaptiveLIFNeuron

from evolutionary.network.snn import TwoLayerSNN, ThreeLayerSNN
from evolutionary.utils.constructors import build_network


def model_to_header(folder, parameters):
    individual_id = "_".join(
        [s.replace(".net", "") for s in parameters.split("/")[-2:]]
    )
    save_folder = folder + f"/saved+{individual_id}/"
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
        # In -> hid
        save_connection_header(network.fc1, "01", save_folder)
        # Hid
        save_neuron_header(network.neuron1, "1", save_folder)
        # Hid -> out
        save_connection_header(network.fc2, "12", save_folder)
        # Out
        save_neuron_header(network.neuron2, "2", save_folder)
        # Network
        save_network_header(
            network, config["net"]["layer sizes"], ["01", "12"], ["1", "2"], save_folder
        )
    # Three-layer
    elif isinstance(network, ThreeLayerSNN):
        # In -> hid1
        save_connection_header(network.fc1, "01", save_folder)
        # Hid1
        save_neuron_header(network.neuron1, "1", save_folder)
        # Hid1 -> hid2
        save_connection_header(network.fc2, "12", save_folder)
        # Hid2
        save_neuron_header(network.neuron2, "2", save_folder)
        # Hid2 -> out
        save_connection_header(network.fc3, "23", save_folder)
        # Out
        save_neuron_header(network.neuron3, "3", save_folder)
        # Network
        save_network_header(
            network,
            config["net"]["layer sizes"],
            ["01", "12", "23"],
            ["1", "2", "3"],
            save_folder,
        )


def save_connection_header(connection, id, save_folder):
    # Get relevant data
    weights = connection.weight.view(-1).tolist()
    post = connection.weight.size(0)
    pre = connection.weight.size(1)

    # Create string
    string = [
        "//Auto-generated",
        '#include "Connection.h"',
        f"float const w_{id}[] = {{{', '.join([str(w) for w in weights])}}};",
        f"ConnectionConf const conf_{id} = {{{post}, {pre}, w_{id}}};",
    ]

    # Write to file
    with open(save_folder + f"connection_conf_{id}.h", "w") as f:
        for line in string:
            f.write(f"{line}\n")


def save_neuron_header(neuron, id, save_folder):
    # Get relevant data
    neuron_type = 1 if isinstance(neuron, AdaptiveLIFNeuron) else 0
    a_v = neuron.alpha_v.view(-1).tolist()
    a_th = (
        neuron.alpha_thresh.view(-1).tolist()
        if isinstance(neuron, AdaptiveLIFNeuron)
        else torch.zeros_like(neuron.alpha_v).view(-1).tolist()
    )
    a_t = neuron.alpha_t.view(-1).tolist()
    d_v = neuron.tau_v.view(-1).tolist()
    d_th = (
        neuron.tau_thresh.view(-1).tolist()
        if isinstance(neuron, AdaptiveLIFNeuron)
        else torch.zeros_like(neuron.tau_v).view(-1).tolist()
    )
    d_t = neuron.tau_t.view(-1).tolist()
    v_rest = neuron.v_rest.item()
    th_rest = (
        (torch.ones_like(neuron.thresh) * neuron.thresh_center).view(-1).tolist()
        if isinstance(neuron, AdaptiveLIFNeuron)
        else neuron.thresh.view(-1).tolist()
    )
    size = neuron.spikes.size(-1)

    # Create string
    string = [
        "//Auto-generated",
        '#include "Neuron.h"',
        f"float const a_v_{id}[] = {{{', '.join([str(a) for a in a_v])}}};",
        f"float const a_th_{id}[] = {{{', '.join([str(a) for a in a_th])}}};",
        f"float const a_t_{id}[] = {{{', '.join([str(a) for a in a_t])}}};",
        f"float const d_v_{id}[] = {{{', '.join([str(d) for d in d_v])}}};",
        f"float const d_th_{id}[] = {{{', '.join([str(d) for d in d_th])}}};",
        f"float const d_t_{id}[] = {{{', '.join([str(d) for d in d_t])}}};",
        f"float const th_rest_{id}[] = {{{', '.join([str(t) for t in th_rest])}}};",
        f"NeuronConf const conf_{id} = {{{neuron_type}, {size}, a_v_{id}, a_th_{id}, a_t_{id}, d_v_{id}, d_th_{id}, d_t_{id}, {v_rest}, th_rest_{id}}};",
    ]

    # Write to file
    with open(save_folder + f"neuron_conf_{id}.h", "w") as f:
        for line in string:
            f.write(f"{line}\n")


def save_network_header(network, layer_sizes, conn_ids, neuron_ids, save_folder):
    # Get data
    if network.encoding == "both":
        encoding_type = 0
    elif network.encoding == "both setpoint":
        encoding_type = 1
    else:
        raise ValueError(f"Incompatible encoding {network.encoding} specified")
    if network.decoding == "weighted trace":
        decoding_type = 0
    else:
        raise ValueError(f"Incompatible decoding {network.decoding} specified")
    setpoint = network.setpoint
    actions = torch.linspace(*network.out_bounds, layer_sizes[-1]).tolist()
    in_size = 2

    # Create string
    conf_string = []
    for i in range(len(conn_ids)):
        conf_string.append(f"&conf_{conn_ids[i]}, &conf_{neuron_ids[i]}, ")
    conf_string[-1] = conf_string[-1][:-2]
    if isinstance(network, TwoLayerSNN):
        string = ["//Auto-generated", '#include "TwoLayerNetwork.h"']
    elif isinstance(network, ThreeLayerSNN):
        string = ["//Auto-generated", '#include "ThreeLayerNetwork.h"']
    else:
        raise ValueError(f"Incompatible network type specified")
    for id in conn_ids:
        string.append(f'#include "connection_conf_{id}.h"')
    for id in neuron_ids:
        string.append(f'#include "neuron_conf_{id}.h"')
    string.append(
        f"float const actions[] = {{{', '.join([str(a) for a in actions])}}};"
    )
    string.append(
        f"NetworkConf const conf = {{{encoding_type}, {decoding_type}, {setpoint}, actions, {in_size}, {', '.join([str(l) for l in layer_sizes])}, {''.join(conf_string)}}};"
    )

    # Write to file
    with open(save_folder + "network_conf.h", "w") as f:
        for line in string:
            f.write(f"{line}\n")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--parameters", type=str, required=True)
    args = vars(parser.parse_args())

    # Call
    model_to_header(args["folder"], args["parameters"])
