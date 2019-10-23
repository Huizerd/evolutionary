from functools import partial

from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN


def build_network(config):
    # Select architecture
    if config["network"] == "ANN":
        network = ANN(2, config["hidden size"], 1)
    elif config["network"] == "SNN":
        if config["double neurons"]:
            inputs = 4
        else:
            inputs = 2
        if config["double actions"]:
            outputs = 2 if config["snn"]["decoding"] != "weighted trace" else 5
        else:
            outputs = 1
        network = SNN(inputs, config["hidden size"], outputs, config)
    else:
        raise KeyError("Not a valid network key!")

    return network


def build_network_partial(config):
    # Select architecture
    if config["network"] == "ANN":
        network = partial(ANN, 2, config["hidden size"], 1)
    elif config["network"] == "SNN":
        if config["double neurons"]:
            inputs = 4
        else:
            inputs = 2
        if config["double actions"]:
            outputs = 2 if config["snn"]["decoding"] != "weighted trace" else 5
        else:
            outputs = 1
        network = partial(SNN, inputs, config["hidden size"], outputs, config)
    else:
        raise KeyError("Not a valid network key!")

    return network
