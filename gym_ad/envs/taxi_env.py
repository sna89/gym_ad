from gym.envs.toy_text.taxi import TaxiEnv


class TaxiEnvV4(TaxiEnv):
    def __init__(self):
        super(TaxiEnvV4, self).__init__()

    @property
    def state(self):
        return self.s

    @state.setter
    def state(self, new_state):
        self.s = new_state
