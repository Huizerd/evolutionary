import torch
import numpy as np


def randomize_env(env, config):
    # Randomize delay, noise, proportional noise, thrust time constant, dt, computational jitter and seed
    env.delay = np.random.randint(*config["env"]["delay"])
    env.noise_std = np.random.uniform(*config["env"]["noise"])
    env.noise_p_std = np.random.uniform(*config["env"]["noise p"])
    env.thrust_tc = np.random.uniform(*config["env"]["thrust tc"])
    env.dt = np.random.uniform(*config["env"]["dt"])
    env.ds_act = np.random.randint(*config["env"]["ds act"])
    env.jitter_prob = np.random.uniform(*config["env"]["jitter"])
    env.seed(np.random.randint(config["env"]["seeds"]))

    # And check values again
    env.checks()

    return env


def getset_env(env, config=None):
    # Set if we have a config, else return current config
    if config is not None:
        env.delay = config["delay"]
        env.noise_std = config["noise"]
        env.noise_p_std = config["noise p"]
        env.thrust_tc = config["thrust tc"]
        env.dt = config["dt"]
        env.ds_act = config["ds act"]
        env.jitter_prob = config["jitter"]
        env.seed(config["seeds"])
        return env
    else:
        config = {}
        config["delay"] = env.delay
        config["noise"] = env.noise_std
        config["noise p"] = env.noise_p_std
        config["thrust tc"] = env.thrust_tc
        config["dt"] = env.dt
        config["ds act"] = env.ds_act
        config["jitter"] = env.jitter_prob
        config["seeds"] = env.seeds
        return config


def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=np.bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # keep any point with a lower cost
            is_efficient[i] = True  # and keep self
    return is_efficient


def sigmoid(x, y_min, y_step, x_mid, steepness):
    y = torch.where(
        x >= 0,
        y_step / (1 + torch.exp(-steepness * (x - x_mid))) + y_min,
        (y_step * torch.exp(steepness * x)) / (1 + torch.exp(steepness * (x - x_mid)))
        + y_min,
    )
    return y
