from gym_quad.envs import QuadHover as QuadBase


class QuadHover(QuadBase):
    def _get_reward(self):
        return abs(-2.0 * self.state[1] / max(1e-5, self.state[0]))
