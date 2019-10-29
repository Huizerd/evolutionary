import numpy as np


def randomize_env(env, config):
    # Randomize delay, noise, proportional noise, thrust time constant, dt, computational jitter and seed
    env.delay = np.random.randint(*config["env"]["delay"])
    env.noise_std = np.random.uniform(*config["env"]["noise"])
    env.noise_p_std = np.random.uniform(*config["env"]["noise p"])
    env.thrust_tc = np.random.uniform(*config["env"]["thrust tc"])
    env.dt = np.random.uniform(*config["env"]["dt"])
    env.jitter_prob = np.random.uniform(*config["env"]["jitter"])
    env.seed(np.random.randint(config["env"]["seeds"]))

    # And check values again
    env.checks()

    return env
