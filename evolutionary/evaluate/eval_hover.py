import torch

from pysnn.network import SNNNetwork

from evolutionary.utils.utils import randomize_env


def eval_hover(config, env, h0, individual):
    # Randomize environment
    env = randomize_env(env, config)

    # Maximize air time, minimize total divergence experienced, minimize offset
    # between initial and final height
    t_score, d_score, h_score = 0.0, 0.0, 0.0

    for h in h0:
        # Reset network and env
        if isinstance(individual[0], SNNNetwork):
            individual[0].reset_state()
        obs = env.reset(h0=h)
        done = False

        while not done:
            obs = torch.from_numpy(obs)
            action = individual[0].forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, div, done, _ = env.step(action)
            # Increment divergence score each step
            d_score += abs(div)

        # Increment other scores
        # Subtract t_score because we use weights of -1.0 for all objectives
        t_score -= env.t
        h_score += abs(h - env.state[0])

    # Decrease importance of divergence score
    # TODO: or do this via weights in DEAP?
    return t_score, d_score / 4, h_score
