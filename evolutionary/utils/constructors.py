from functools import partial

import torch

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
            outputs = 2
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
            outputs = 2
        else:
            outputs = 1
        network = partial(SNN, inputs, config["hidden size"], outputs, config)
    else:
        raise KeyError("Not a valid network key!")

    return network


def build_individual(container, config):
    # Select architecture
    if config["network"] == "ANN":
        # Returns generator
        return container(ANN(2, config["hidden size"], 1) for _ in range(1))
    elif config["network"] == "SNN":
        if config["double neurons"]:
            inputs = 4
        else:
            inputs = 2
        if config["double actions"]:
            outputs = 2
        else:
            outputs = 1
        # Returns generator
        return container(
            SNN(inputs, config["hidden size"], outputs, config) for _ in range(1)
        )
    else:
        raise KeyError("Not a valid network key!")


def build_jit_individual(container, config):
    # Select architecture
    if config["network"] == "ANN":
        # Returns generator
        return container(
            torch.jit.script(ANN(2, config["hidden size"], 1)) for _ in range(1)
        )
    elif config["network"] == "SNN":
        raise NotImplementedError("JIT scripting not yet implemented for SNNs")
    else:
        raise KeyError("Not a valid network key!")
