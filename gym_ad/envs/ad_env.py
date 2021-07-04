import gym
from gym.utils import seeding
from config import Keyword
from gym.envs.toy_text.discrete import categorical_sample
from colorama import Style, Fore
import numpy as np
from dataclasses import dataclass
# from typing import List
import itertools


@dataclass
class GymAdState:
    temperature: int = 0
    steps_from_alert: int = 0


#    history: List


class AdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        self.max_temp = config["env"]["max_temp"]
        self.alert_prediction_steps = config["env"]["alert_prediction_steps"]
        self._max_steps_from_alert = self.alert_prediction_steps + 1

        self.reward_false_alert = config["AD_V1"]["reward"]["false_alert"]
        self.reward_missed_alert = config["AD_V1"]["reward"]["missed_alert"]
        self.reward_good_alert = config["AD_V1"]["reward"]["good_alert"]

        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        self.P = self._create_transition()

        self.np_random = None
        self.last_action = None

        self.current_state = (self.max_temp // 2 + 1, self.max_steps_from_alert)
        self.seed()
        self.reset()

    @property
    def max_steps_from_alert(self):
        return self._max_steps_from_alert

    @property
    def min_steps_from_alert(self):
        return 1

    def _create_observation_space(self):
        raise NotImplementedError

    def _create_action_space(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        assert (1 < self.current_state[1] < self.max_steps_from_alert and action == 0) \
               or self.current_state[1] == 1 \
               or self.current_state[1] == self.max_steps_from_alert, \
            'Must choose wait action if alert is triggered'

        self.last_action = action
        transitions = self.P[self.current_state][action]

        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, self.current_state, reward, done = transitions[i]

        return self.current_state, reward, done, prob

    def reset(self):
        self.current_state = (self.max_temp // 2 + 1, self.max_steps_from_alert)
        return self._get_obs()

    def close(self):
        pass

    def _get_obs(self):
        return self.current_state

    def _init_transition_matrix(self):
        num_states_steps_from_alert = self.observation_space[Keyword.STEPS_FROM_ALERT].n
        num_states_temp = self.observation_space[Keyword.TEMPERATURE].n
        num_actions = self.action_space.n

        P = {(c_temp, step): {
            action: [] for action in range(num_actions)
        }
            for c_temp, step in itertools.product(range(num_states_temp), range(1, num_states_steps_from_alert + 1))
        }

        return P

    def _create_transition(self):
        raise NotImplementedError

    @staticmethod
    def _append_to_transition(P, curr_temp, steps_from_alert, action, prob, next_state, reward, done):

        P[(int(curr_temp), int(steps_from_alert))][action].append(
            (
                prob,
                next_state,
                reward,
                done
            )
        )

    def _get_good_alert_reward(self, steps_from_alert):
        return self.reward_good_alert * (self.max_steps_from_alert - steps_from_alert)

    def render(self, mode='human'):
        desc = np.arange(0, self.max_temp + 1, 1).tolist()

        if self.current_state[1] is not None:
            steps = "-"
            if self.current_state[1] < self.max_steps_from_alert:
                steps = self.current_state[1]
            print("Steps from alert: {}".format(steps))

        for i in desc:
            if i == self.current_state[0]:
                print(f'{Fore.RED}{desc[i]}{Style.RESET_ALL}', end=" ")
            else:
                print(desc[i], end=" ")

            if i == self.max_temp:
                print()
