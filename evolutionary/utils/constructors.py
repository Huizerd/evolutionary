from functools import partial

from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN
from evolutionary.environment.environment import QuadEnv


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


def build_environment(config):
    # Where a range was given for randomization, take the lower bound
    # For evaluations, envs are randomized anyway beforehand, and for test we want
    # some determinism
    env = QuadEnv(
        delay=config["env"]["delay"][0],
        noise=config["env"]["noise"][0],
        noise_p=config["env"]["noise p"][0],
        thrust_bounds=config["env"]["thrust bounds"],
        thrust_tc=config["env"]["thrust tc"][0],
        settle=config["env"]["settle"],
        wind=config["env"]["wind"],
        h0=config["env"]["h0"][0],
        dt=config["env"]["dt"][0],
        jitter=config["env"]["jitter"][0],
        max_t=config["env"]["max time"],
        seed=None,
    )
    return env
