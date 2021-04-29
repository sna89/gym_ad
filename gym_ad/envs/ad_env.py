import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from io import StringIO
import sys
from contextlib import closing


class AdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.max_temp = 10
        self.num_alert_prediction_steps = 3

        self.prob = [float(1) / 3] * 3
        self.temp_delta = [-1, 0, 1]

        self.observation_space = spaces.Dict({
            "temperature": spaces.Discrete(self.max_temp + 1),
            # num_temp_levels + temp_lower_threshold (0)
            "steps_from_alert": spaces.Discrete(self.num_alert_prediction_steps + 2)
            # num_steps (1, 2, .. n) + no_alert (n + 1) + alert_time (0)
        })
        self.action_space = spaces.Discrete(2)  # wait , alert

        self.current_temp = None
        self.steps_from_alert = None

        self.last_action = None
        self.last_delta = None

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.steps_from_alert == self.num_alert_prediction_steps + 1 and action == 1, \
            'Must choose wait action if alert is triggered'

        temp_delta_next_step = np.random.choice(self.temp_delta, size=1, p=self.prob)
        self._update_current_temp(temp_delta_next_step)

        reward = 0
        done = False

        if action:  # alert
            self.steps_from_alert = self.num_alert_prediction_steps
            if self.current_temp == 0:
                reward, done = self._missed_alert()

        else:  # wait
            if self.steps_from_alert == self.num_alert_prediction_steps + 1:  # no triggered alert
                if self.current_temp == 0:
                    reward, done = self._missed_alert()

            elif self.num_alert_prediction_steps + 1 > self.steps_from_alert > 0:  # Active triggered alert
                if self.current_temp == 0:
                    reward, done = self._catched_alert()
                self.steps_from_alert -= 1

            elif self.steps_from_alert == 0:  # Active triggered alert - last time step
                if self.current_temp == 0:
                    reward, done = self._catched_alert()

                elif self.current_temp > 0:
                    reward, done = self._false_alert()

                self.steps_from_alert = self.num_alert_prediction_steps + 1

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.current_temp = self.max_temp
        self.steps_from_alert = self.num_alert_prediction_steps + 1
        return

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_obs(self):
        return {"temperature": self.current_temp,
                "steps_from_alert": self.steps_from_alert}

    def _update_current_temp(self, temp_delta_next_step):
        self.current_temp += temp_delta_next_step
        if self.current_temp > self.max_temp:
            self.current_temp = self.max_temp

        if self.current_temp < 0:
            self.current_temp = 0

    def _missed_alert(self):
        self.reset()
        return -100, True

    def _catched_alert(self):
        self.reset()
        return 10, True

    @staticmethod
    def _false_alert():
        return -10, False
