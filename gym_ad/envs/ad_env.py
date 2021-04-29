import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from colorama import Style, Fore
from constants import Constants, Reward, Keyword


class AdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.max_temp = Constants.MAX_TEMP
        self.desc = np.arange(0, self.max_temp + 1, 1).tolist()
        self.num_alert_prediction_steps = Constants.ALERT_PREDICTION_STEPS

        self.prob = [float(1) / 3] * 3
        self.temp_delta = [-1, 0, 1]

        self.observation_space = spaces.Dict({
            Keyword.TEMPERATURE: spaces.Discrete(self.max_temp + 1),
            # num_temp_levels + temp_lower_threshold (0)
            Keyword.STEPS_FROM_ALERT: spaces.Discrete(self.num_alert_prediction_steps + 2)
            # num_steps (1, 2, .. n) + no_alert (n + 1) + alert_time (0)
        })
        self.action_space = spaces.Discrete(2)  # wait , alert
        self.P = self._create_transition()

        self.current_temp = None
        self.steps_from_alert = None

        self.last_action = None
        self.last_delta = None

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.steps_from_alert == self.num_alert_prediction_steps + 1 and action == 0, \
            'Must choose wait action if alert is triggered'

        self.last_delta = np.random.choice(self.temp_delta, size=1, p=self.prob)
        self._update_current_temp()
        self.last_action = action

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
                    reward, done = self._good_alert()
                self.steps_from_alert -= 1

            elif self.steps_from_alert == 0:  # Active triggered alert - last time step
                if self.current_temp == 0:
                    reward, done = self._good_alert()

                elif self.current_temp > 0:
                    reward, done = self._false_alert()

                self.steps_from_alert = self.num_alert_prediction_steps + 1

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.current_temp = self.max_temp
        self.steps_from_alert = self.num_alert_prediction_steps + 1
        return self._get_obs()

    def render(self, mode='human'):
        if self.last_action in [0, 1]:
            print("Last Action: {}".format(["Wait", "Alert"][self.last_action]))
        if self.last_delta in [-1, 0, 1]:
            print("Last Delta: {}".format(self.last_delta))
        for i in self.desc:
            if i == self.current_temp:
                print(f'{Fore.RED}{self.desc[i]}{Style.RESET_ALL}', end=" ")
            else:
                print(self.desc[i], end=" ")

            if i == self.max_temp:
                print()

    def close(self):
        pass

    def _get_obs(self):
        return {Keyword.TEMPERATURE: self.current_temp,
                Keyword.STEPS_FROM_ALERT: self.steps_from_alert}

    def _update_current_temp(self):
        self.current_temp += self.last_delta
        if self.current_temp > self.max_temp:
            self.current_temp = self.max_temp

        if self.current_temp < 0:
            self.current_temp = 0

    def _missed_alert(self):
        self.reset()
        return Reward.MISSED_ALERT, True

    def _good_alert(self):
        self.reset()
        return Reward.GOOD_ALERT, True

    @staticmethod
    def _false_alert():
        return Reward.FALSE_ALERT, False

    def _create_transition(self):
        num_states_steps = self.observation_space[Keyword.STEPS_FROM_ALERT].n
        num_states_temp = self.observation_space[Keyword.TEMPERATURE].n

        P = {curr_temp: {
            steps: [] for steps in range(num_states_steps)
        } for curr_temp in range(num_states_temp)}

        for curr_temp in range(num_states_temp):
            for steps_to_alert in range(num_states_steps):

                if steps_to_alert == num_states_steps:
                    if 1 < curr_temp < self.max_temp:
                        for i in range(len(self.prob)):
                            P[curr_temp][steps_to_alert].append(
                                (self.prob[i],
                                 {Keyword.TEMPERATURE: curr_temp + self.temp_delta[i],
                                  Keyword.STEPS_FROM_ALERT: self.steps_from_alert
                                  },
                                 0,
                                 False)
                            )

                    if curr_temp == 1:
                        P[curr_temp][steps_to_alert].append(
                            (self.prob[0],
                             {Keyword.TEMPERATURE: curr_temp + self.temp_delta[0],
                              Keyword.STEPS_FROM_ALERT: self.steps_from_alert
                              },
                             Reward.MISSED_ALERT,
                             True)
                        )

                        for i in range(1, 3):
                            P[curr_temp][steps_to_alert].append(
                                (self.prob[i],
                                 {Keyword.TEMPERATURE: curr_temp + self.temp_delta[i],
                                  Keyword.STEPS_FROM_ALERT: self.steps_from_alert
                                  },
                                 0,
                                 False)
                            )

                    if curr_temp == 0:
                        P[curr_temp][steps_to_alert].append(
                            (1,
                             {Keyword.TEMPERATURE: self.max_temp,
                              Keyword.STEPS_FROM_ALERT: self.steps_from_alert
                              },
                             0,
                             False)
                        )

                    if curr_temp == self.max_temp:
                        P[curr_temp][steps_to_alert].append(
                            (self.prob[0],
                             {
                                 Keyword.TEMPERATURE: self.max_temp + self.temp_delta[0]
                                 Keyword.STEPS_FROM_ALERT: self.steps_from_alert
                             },
                             0,
                             False)
                        )
                        P[curr_temp][steps_to_alert].append(
                            (
                                self.prob[0] + self.prob[1],
                                {
                                    Keyword.TEMPERATURE: self.max_temp,
                                    Keyword.STEPS_FROM_ALERT: self.steps_from_alert
                                }
                            )
                        )
                else:


        return 0
