import numpy as np

from gym_quad.envs.quad_hover import QuadHover


# Test constant-divergence landings with proportional control
# Divergence setpoint = 0.5, P-gain = 0.01
def test_D_landings():
    env = QuadHover(
        h0=3.0,
        delay=2,
        noise=0.1,
        noise_p=0.1,
        wind=0.1,
        seed=1,
        g_bounds=(-0.5, 0.5),
        min_h=0.1,
        settle=0.1,
    )

    obs = env.reset(h0=3.0)
    done = False

    t = [env.t]
    h = [env.state[0]]
    while not done:
        obs, _, done, _ = env.step((obs[..., 0] - 0.5).clip(-0.5, 0.5))
        t.append(env.t)
        h.append(env.state[0])

    import matplotlib.pyplot as plt

    plt.plot(t, h)
    plt.show()


if __name__ == "__main__":
    test_D_landings()
