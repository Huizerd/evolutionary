from functools import partial

from evolutionary.network.ann import ANN
from evolutionary.network.snn import SNN
from evolutionary.environment.environment import QuadEnv


def build_network(config):
    # Select architecture
    if config["net"]["network"] == "ANN":
        network = ANN(config)
    elif config["net"]["network"] == "SNN":
        network = SNN(config)
    else:
        raise ValueError("Not a valid network")

    return network


def build_network_partial(config):
    # Select architecture
    if config["net"]["network"] == "ANN":
        network = partial(ANN, config)
    elif config["net"]["network"] == "SNN":
        network = partial(SNN, config)
    else:
        raise ValueError("Not a valid network")

    return network


def build_environment(config):
    # Where a range was given for randomization, take the lower bound
    # For evaluations, envs are randomized anyway beforehand, and for test we want
    # some determinism
    env = QuadEnv(
        delay=config["env"]["delay"][0],
        noise=config["env"]["noise"][0],
        noise_p=config["env"]["noise p"][0],
        g=config["env"]["g"],
        g_bounds=config["env"]["g bounds"],
        thrust_tc=config["env"]["thrust tc"][0],
        settle=config["env"]["settle"],
        wind=config["env"]["wind"],
        h0=config["env"]["h0"][0],
        dt=config["env"]["dt"][0],
        ds_act=config["env"]["ds act"][0],
        jitter=config["env"]["jitter"][0],
        max_t=config["env"]["max time"],
        seed=None,
    )
    return env
