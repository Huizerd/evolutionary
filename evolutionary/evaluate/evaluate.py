import torch

from pysnn.network import SNNNetwork

from evolutionary.utils.utils import randomize_env


def evaluate(config, env, h0, individual):
    # Randomize environment
    env = randomize_env(env, config)

    # All possible objectives: air time, time to land, final height, final offset,
    # final offset from 5 m, final velocity, unsigned divergence, signed divergence
    objectives = {
        "air time": 0.0,
        "time to land": 0.0,
        "final height": 0.0,
        "final offset": 0.0,
        "final offset 5m": 0.0,
        "final velocity": 0.0,
        "unsigned divergence": 0.0,
        "signed divergence": 0.0,
    }

    for h in h0:
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
        # Air time and time to land
        if env.t >= env.MAX_T:
            objectives["air time"] += env.MAX_T
            objectives["time to land"] += env.MAX_T
        else:
            objectives["air time"] += env.t
            objectives["time to land"] += env.t

        # Final height and final offset
        objectives["final height"] += env.state[0]
        objectives["final offset"] += abs(h - env.state[0])
        objectives["final offset 5m"] += abs(5.0 - env.state[0])

        # Final velocity
        # TODO: check whether this makes much of a difference with Kirk's approach
        # TODO: he had 4.0 for not landing, else env.state[1]**2
        objectives["final velocity"] += env.state[1] * env.state[1]

    # Select appropriate objectives
    # List, so order is guaranteed
    assert len(config["evo"]["objectives"]) == 3, "Only 3 objectives are supported"
    assert all(
        [obj in objectives for obj in config["evo"]["objectives"]]
    ), "Invalid objective"
    return [objectives[obj] for obj in config["evo"]["objectives"]]
