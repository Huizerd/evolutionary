import torch

from pysnn.network import SNNNetwork

from evolutionary.utils.utils import randomize_env


def evaluate(valid_objectives, config, envs, h0, individual):
    # Keep track of all possible objectives
    objectives = {obj: 0.0 for obj in valid_objectives}

    for h, env in zip(h0, envs):
        # Reset network and env
        if isinstance(individual[0], SNNNetwork):
            individual[0].reset_state()
        obs = env.reset(h0=h)
        done = False
        spikes = 0

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = individual[0].forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)
            # Increment number of spikes each step
            if isinstance(individual[0], SNNNetwork):
                spikes += (
                    individual[0].neuron1.spikes.sum().item()
                    + individual[0].neuron2.spikes.sum().item()
                    if individual[0].neuron1 is not None
                    else individual[0].neuron2.spikes.sum().item()
                )

        # Increment other scores
        # Time to land, final height and final velocity
        if env.t >= env.max_t or env.state[0] >= env.MAX_H:
            objectives["time to land"] += 100.0
            objectives["time to land scaled"] += 100.0
            objectives["final velocity"] += 10.0
            objectives["final velocity squared"] += 10.0
            objectives["final height"] += 10.0
        else:
            objectives["time to land"] += env.t - config["env"]["settle"]
            objectives["time to land scaled"] += (env.t - config["env"]["settle"]) / h
            objectives["final velocity"] += abs(env.state[1])
            objectives["final velocity squared"] += env.state[1] ** 2
            objectives["final height"] += env.state[0]

        # Spikes divided by real time to land, because we don't want to overly stimulate
        # too fast landings
        objectives["spikes"] += spikes / (env.t - config["env"]["settle"])

    # Select appropriate objectives
    # List, so order is guaranteed
    return [objectives[obj] / len(h0) for obj in config["evo"]["objectives"]]
