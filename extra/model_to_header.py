import argparse
import os
import yaml

import torch

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
    size = neuron.spikes.size(-1)
    v_rest = neuron.v_rest.item()
    a_t = neuron.alpha_t.view(-1).tolist()
    # This works
    d_v = ((4096 - neuron.tau_v.view(-1)) / 4096).tolist()
    d_t = neuron.tau_t.view(-1).tolist()
    th = neuron.thresh.view(-1).tolist()

    # Create string
    string = [
        "//Auto-generated",
        '#include "Neuron.h"',
        f"float const a_t_{id}[] = {{{', '.join([str(a) for a in a_t])}}};",
        f"float const d_v_{id}[] = {{{', '.join([str(d) for d in d_v])}}};",
        f"float const d_t_{id}[] = {{{', '.join([str(d) for d in d_t])}}};",
        f"float const th_{id}[] = {{{', '.join([str(t) for t in th])}}};",
        f"NeuronConf const conf_{id} = {{{size}, a_t_{id}, d_v_{id}, d_t_{id}, {v_rest}, th_{id}}};",
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
    elif network.encoding == "cubed-spike place":
        encoding_type = 2
    else:
        raise ValueError(f"Incompatible encoding {network.encoding} specified")
    if network.decoding == "weighted trace":
        decoding_type = 0
    else:
        raise ValueError(f"Incompatible decoding {network.decoding} specified")
    setpoint = network.setpoint
    centers = network.buckets.tolist()
    actions = network.trace_weights.tolist()
    in_size = 2

    # Create string
    conf_string = []
    for i in range(len(conn_ids)):
        conf_string.append(f"&conf_{conn_ids[i]}, &conf_{neuron_ids[i]}, ")
    conf_string[-1] = conf_string[-1][:-2]
    string = ["//Auto-generated", '#include "TwoLayerNetwork.h"']
    for id in conn_ids:
        string.append(f'#include "connection_conf_{id}.h"')
    for id in neuron_ids:
        string.append(f'#include "neuron_conf_{id}.h"')
    string.append(
        f"float const centers[] = {{{', '.join([str(c) for c in centers])}}};"
    )
    string.append(
        f"float const actions[] = {{{', '.join([str(a) for a in actions])}}};"
    )
    string.append(
        f"NetworkConf const conf = {{{encoding_type}, {decoding_type}, {setpoint}, centers, actions, {in_size}, {', '.join([str(l) for l in layer_sizes])}, {''.join(conf_string)}}};"
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
