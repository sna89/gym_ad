import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from colorama import Style, Fore
from constants import Constants, Reward, Keyword
from gym.envs.toy_text.discrete import categorical_sample


class AdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.max_temp = Constants.MAX_TEMP
        self.max_steps_from_alert = Constants.ALERT_PREDICTION_STEPS + 1

        self.desc = np.arange(0, self.max_temp + 1, 1).tolist()
        self.num_alert_prediction_steps = Constants.ALERT_PREDICTION_STEPS

        self.prob_list = [float(1) / 3] * 3
        self.temperature_delta_list = [-1, 0, 1]

        self.observation_space = spaces.Dict({
            Keyword.TEMPERATURE: spaces.Discrete(self.max_temp + 1),
            # num_temp_levels + temp_lower_threshold (0)
            Keyword.STEPS_FROM_ALERT: spaces.Discrete(self.num_alert_prediction_steps + 2)
            # num_steps (1, 2, .. n) + no_alert (n + 1) + alert_time (0)
        })
        self.action_space = spaces.Discrete(2)  # wait , alert

        self.P = self._create_transition()

        self.np_random = None
        self.last_action = None

        self.current_state = dict()
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        assert (0 < self.current_state[Keyword.STEPS_FROM_ALERT] < self.max_steps_from_alert and action == 0)\
            or self.current_state[Keyword.STEPS_FROM_ALERT] == 0\
            or self.current_state[Keyword.STEPS_FROM_ALERT] == self.max_steps_from_alert, \
            'Must choose wait action if alert is triggered'

        self.last_action = action
        transitions = self.P[self.current_state[Keyword.TEMPERATURE]][self.current_state[Keyword.STEPS_FROM_ALERT]][action]

        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, self.current_state, reward, done = transitions[i]

        return self.current_state, reward, done, {}

    def reset(self):
        self.current_state[Keyword.TEMPERATURE] = self.max_temp
        self.current_state[Keyword.STEPS_FROM_ALERT] = self.max_steps_from_alert
        return self._get_obs()

    def render(self, mode='human'):
        if self.last_action is not None:
            print("Last Action: {}".format(["Wait", "Alert"][self.last_action]))

        if self.current_state[Keyword.STEPS_FROM_ALERT] is not None:
            print("Steps from alert: {}".format(self.current_state[Keyword.STEPS_FROM_ALERT]))

        for i in self.desc:
            if i == self.current_state[Keyword.TEMPERATURE]:
                print(f'{Fore.RED}{self.desc[i]}{Style.RESET_ALL}', end=" ")
            else:
                print(self.desc[i], end=" ")

            if i == self.max_temp:
                print()
                print()

    def close(self):
        pass

    def _get_obs(self):
        return self.current_state

    def _create_transition(self):
        num_states_steps_from_alert = self.observation_space[Keyword.STEPS_FROM_ALERT].n
        num_states_temp = self.observation_space[Keyword.TEMPERATURE].n
        num_actions = self.action_space.n

        P = {curr_temp: {
            step: {
                action: [] for action in range(num_actions)
            } for step in range(num_states_steps_from_alert)
        } for curr_temp in range(num_states_temp)}

        for curr_temp in range(num_states_temp):
            for steps_from_alert in range(num_states_steps_from_alert):
                for action in range(num_actions):

                    if curr_temp == self.max_temp:
                        reward = 0
                        done = False

                        delta_step_from_alert = None

                        if steps_from_alert == self.max_steps_from_alert:
                            if action == 0:
                                delta_step_from_alert = 0
                            else:
                                delta_step_from_alert = -1

                        elif 1 < steps_from_alert < self.max_steps_from_alert:
                            if action == 0:
                                delta_step_from_alert = -1
                            else:
                                continue

                        elif steps_from_alert == 1:
                            if action == 0:
                                delta_step_from_alert = -1
                                reward = Reward.FALSE_ALERT
                            else:
                                continue

                        elif steps_from_alert == 0:
                            if action == 0:
                                delta_step_from_alert = self.max_steps_from_alert
                            else:
                                delta_step_from_alert = self.max_steps_from_alert - 1

                        for i in range(len(self.prob_list) - 1):
                            temp_delta = self.temperature_delta_list[i]

                            if temp_delta == -1:
                                prob = self.prob_list[i] + self.prob_list[i + 1]
                            else:
                                prob = self.prob_list[i]
                                temp_delta = 0

                            self._append_to_transition(P,
                                                       prob=prob,
                                                       curr_temperature=curr_temp,
                                                       delta_temperature=temp_delta,
                                                       curr_steps_from_alert=steps_from_alert,
                                                       delta_steps_from_alert=delta_step_from_alert,
                                                       action=action,
                                                       reward=reward,
                                                       done=done)

                    elif 1 < curr_temp < self.max_temp:
                        reward = Reward.FALSE_ALERT if steps_from_alert == 1 else 0
                        delta_step_from_alert = 0

                        if steps_from_alert == self.max_steps_from_alert:
                            if action == 0:
                                delta_step_from_alert = 0
                            else:
                                delta_step_from_alert = -1

                        elif 1 <= steps_from_alert < self.max_steps_from_alert:
                            if action == 0:
                                delta_step_from_alert = -1
                            else:
                                continue

                        elif steps_from_alert == 0:
                            if action == 0:
                                delta_step_from_alert = self.max_steps_from_alert
                            else:
                                delta_step_from_alert = self.max_steps_from_alert - 1

                        for i in range(len(self.prob_list)):
                            self._append_to_transition(P,
                                                       prob=self.prob_list[i],
                                                       curr_temperature=curr_temp,
                                                       delta_temperature=self.temperature_delta_list[i],
                                                       curr_steps_from_alert=steps_from_alert,
                                                       delta_steps_from_alert=delta_step_from_alert,
                                                       action=action,
                                                       reward=reward,
                                                       done=False)

                    elif curr_temp == 1:
                        delta_step_from_alert = None

                        for i in range(len(self.prob_list)):
                            temp_delta = self.temperature_delta_list[i]
                            done = True if temp_delta == -1 else False
                            reward = 0

                            if steps_from_alert == self.max_steps_from_alert:
                                if action == 0:
                                    delta_step_from_alert = 0
                                    if temp_delta == -1:
                                        reward = Reward.MISSED_ALERT
                                else:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = Reward.GOOD_ALERT * (steps_from_alert + 1)
                                        delta_step_from_alert = 0

                            elif 1 < steps_from_alert < self.max_steps_from_alert:
                                if action == 0:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = Reward.GOOD_ALERT * (steps_from_alert + 1)
                                else:
                                    continue

                            elif steps_from_alert == 1:
                                if action == 0:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = Reward.GOOD_ALERT * (steps_from_alert + 1)
                                    else:
                                        reward = Reward.FALSE_ALERT
                                else:
                                    continue

                            elif steps_from_alert == 0:
                                if action == 0:
                                    delta_step_from_alert = self.max_steps_from_alert
                                    if temp_delta == -1:
                                        reward = Reward.MISSED_ALERT
                                else:
                                    delta_step_from_alert = self.max_steps_from_alert - 1
                                    if temp_delta == -1:
                                        reward = Reward.GOOD_ALERT * (steps_from_alert + 1)

                            self._append_to_transition(P,
                                                       prob=self.prob_list[i],
                                                       curr_temperature=curr_temp,
                                                       delta_temperature=temp_delta,
                                                       curr_steps_from_alert=steps_from_alert,
                                                       delta_steps_from_alert=delta_step_from_alert,
                                                       action=action,
                                                       reward=reward,
                                                       done=done)

                    elif curr_temp == 0:
                        if action == 0:
                            temp_delta = self.max_temp
                            reward = 0
                            done = False

                            delta_step_from_alert = self.max_steps_from_alert - steps_from_alert

                            self._append_to_transition(P,
                                                       prob=1,
                                                       curr_temperature=curr_temp,
                                                       delta_temperature=temp_delta,
                                                       curr_steps_from_alert=steps_from_alert,
                                                       delta_steps_from_alert=delta_step_from_alert,
                                                       action=action,
                                                       reward=reward,
                                                       done=done)
                        else:
                            continue

        return P

    def _append_to_transition(self,
                              P,
                              prob,
                              curr_temperature,
                              delta_temperature,
                              curr_steps_from_alert,
                              delta_steps_from_alert,
                              action,
                              reward,
                              done
                              ):

        next_temperature = curr_temperature + delta_temperature
        next_step_from_alert = curr_steps_from_alert + delta_steps_from_alert

        next_state = {Keyword.TEMPERATURE: next_temperature,
                      Keyword.STEPS_FROM_ALERT: next_step_from_alert}

        self._append_to_transition_inner(P, curr_temperature, curr_steps_from_alert, action, prob,
                                         next_state, reward, done)

    @staticmethod
    def _append_to_transition_inner(P, curr_temp, steps_from_alert, action, prob, next_state, reward, done):
        P[curr_temp][steps_from_alert][action].append(
            (
                prob,
                next_state,
                reward,
                done
            )
        )
