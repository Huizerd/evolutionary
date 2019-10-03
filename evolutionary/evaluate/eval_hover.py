import numpy as np
import torch


def eval_hover(env, h0, individual):
    # Maximize air time, minimize total divergence experienced, minimize offset
    # between initial and final height
    t_score, d_score, h_score = 0.0, 0.0, 0.0

    for h in h0:
        obs = env.reset(h0=h)
        done = False

        while not done:
            obs = torch.from_numpy(obs)
            action = individual[0].forward(obs.view(1, 1, -1))
            action = action.numpy()
            obs, div, done, _ = env.step(action)
            # Increment divergence score each step
            d_score += div

        # Increment other scores
        # Subtract t_score because we use weights of -1.0 for all objectives
        t_score -= env.t
        h_score += np.abs(h - env.state[0])

    return t_score, d_score, h_score
