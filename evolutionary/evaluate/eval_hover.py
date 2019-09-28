import numpy as np
import torch

from gym_quad.envs import QuadHover


def eval_hover(individual, seed=None):
    # TODO: make parameters configurable
    # TODO: keep seed in case we want to implement determinism later
    # TODO: is passing seeds down still random in the correct sense?
    np.random.seed(seed)

    env = QuadHover(
        delay=np.random.randint(1, 6),
        comp_delay_prob=0.0,
        noise=np.random.uniform(0.0, 0.15),
        noise_p=np.random.uniform(0.0, 0.25),
        thrust_tc=np.random.uniform(0.02, 0.1),
        settle=1.0,
        wind=0.0,
        h0=5.0,
        dt=0.02,
        seed=np.random.randint(100),
    )

    # Evaluate for various starting heights
    h0 = [2.0, 4.0, 6.0, 8.0]

    # Minimize three scores: time, height and velocity
    t_score, h_score, v_score = 0.0, 0.0, 0.0

    for h in h0:
        obs = env.reset(h0=h)
        done = False

        while not done:
            # TODO: reward is not being used
            obs = torch.from_numpy(obs)
            action = individual[0].forward(obs)
            action = action.numpy()
            obs, _, done, _ = env.step(action)

        # Increment scores
        if env.t >= env.MAX_T or env.state[0] >= env.MAX_H:
            # Penalize not landing
            # TODO: incorporate all this in rewards, let the env give out a tuple of 3?
            t_score += env.MAX_T
            h_score += env.state[0]
            v_score += -2.0 * -2.0
        else:
            t_score += env.t
            h_score += env.state[0]
            v_score += env.state[1] * env.state[1]

    return t_score, h_score, v_score
