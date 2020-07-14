import torch

from evolutionary.utils.utils import randomize_env


def evaluate(config, envs, h0, individual):
    # Keep track of objectives
    d_error = 0.0

    # Go over altitudes
    for h, env in zip(h0, envs):
        # Reset network and env
        individual[0].reset_state()
        obs = env.reset(h0=h)
        done = False

        while not done:
            # Step the environment
            obs = torch.from_numpy(obs)
            action = individual[0].forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, _, done, _ = env.step(action)

            # Increment divergence SSE
            if env.t > config["env"]["settle"]:
                d_error += (config["evo"]["D setpoint"] - env.div_ph[0]) ** 2

    return [d_error / len(h0)]
