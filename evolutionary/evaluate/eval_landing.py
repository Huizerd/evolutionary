import torch

from pysnn.network import SNNNetwork

from evolutionary.utils.utils import randomize_env


def eval_landing(config, env, h0, individual):
    # Randomize environment
    env = randomize_env(env, config)

    # Minimize three scores: time, height and velocity
    # More specifically: total time to land, final height, final velocity
    t_score, h_score, v_score = 0.0, 0.0, 0.0

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
            obs, _, done, _ = env.step(action)

        # Increment scores
        if env.t >= env.MAX_T or env.state[0] >= env.MAX_H:
            # Penalize not landing
            t_score += env.MAX_T
            h_score += env.state[0]
            v_score += -2.0 * -2.0
        else:
            t_score += env.t
            h_score += env.state[0]
            v_score += env.state[1] * env.state[1]

    return t_score, h_score, v_score
