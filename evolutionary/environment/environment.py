from gym_quad.envs import QuadHover as QuadBase


class QuadEnv(QuadBase):
    def _get_reward(self):
        # Divergence might be needed for objectives
        return -2.0 * self.state[1] / max(1e-5, self.state[0])
