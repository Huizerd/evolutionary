import torch

from pysnn.network import SNNNetwork

from evolutionary.utils.utils import randomize_env


def evaluate(valid_objectives, config, env, h0, individual):
    # Keep track of all possible objectives
    objectives = {obj: 0.0 for obj in valid_objectives}

    for h in h0:
        # Randomize environment for each altitude to get better generalization
        env = randomize_env(env, config)
        # Reset network and env
        if isinstance(individual[0], SNNNetwork):
            individual[0].reset_state()
        obs = env.reset(h0=h)
        done = False

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = individual[0].forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, div, done, _ = env.step(action)
            # Increment (un)signed divergence scores each step
            objectives["unsigned divergence"] += abs(div)
            objectives["signed divergence"] += div

        # Increment other scores
        # Air time
        if env.t >= env.max_t:
            objectives["air time"] += env.max_t
        else:
            objectives["air time"] += env.t

        # Time to land and final velocity
        if env.t >= env.max_t or env.state[0] >= env.MAX_H:
            objectives["time to land"] += 100.0
            objectives["final velocity"] += 10.0
            objectives["final velocity squared"] += 10.0
            objectives["final height"] += 10.0
        else:
            objectives["time to land"] += env.t - config["env"]["settle"]
            objectives["final velocity"] += abs(env.state[1])
            objectives["final velocity squared"] += env.state[1] * env.state[1]
            objectives["final height"] += env.state[0]

        # Final height and final offset
        objectives["final offset"] += abs(h - env.state[0])
        objectives["final offset 5m"] += abs(5.0 - env.state[0])

        # Signed divergence should be taken absolute now, since we want to minimize it
        objectives["signed divergence"] = abs(objectives["signed divergence"])

    # Select appropriate objectives
    # List, so order is guaranteed
    return [objectives[obj] for obj in config["evo"]["objectives"]]
