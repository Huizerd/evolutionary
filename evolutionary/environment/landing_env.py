from gym_quad.envs import QuadHover as QuadBase


class QuadLanding(QuadBase):
    def _get_reward(self):
        # We don't use reward in evaluation, but might come in handy later
        pass
