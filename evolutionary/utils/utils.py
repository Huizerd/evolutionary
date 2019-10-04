import dask.bag as db
import numpy as np


def dask_map(func, iterable):
    bag = db.from_sequence(iterable).map(func)
    return bag.compute()


def randomize_env(env, config):
    # Randomize delay, noise, proportional noise, thrust time constant and seed
    env.delay = np.random.randint(*config["env"]["delay"])
    env.noise = np.random.uniform(*config["env"]["noise"])
    env.noise_p = np.random.uniform(*config["env"]["noise p"])
    env.thrust_tc = np.random.uniform(*config["env"]["thrust tc"])
    env.seed(np.random.randint(config["env"]["seeds"]))

    return env
