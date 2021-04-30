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
        self.max_steps_from_alert = Constants.ALERT_PREDICTION_STEPS + 1

        self.desc = np.arange(0, self.max_temp + 1, 1).tolist()
        self.num_alert_prediction_steps = Constants.ALERT_PREDICTION_STEPS

        self.prob = [float(1) / 3] * 3
        self.temperature_delta = [-1, 0, 1]

        self.observation_space = spaces.Dict({
            Keyword.TEMPERATURE: spaces.Discrete(self.max_temp + 1),
            # num_temp_levels + temp_lower_threshold (0)
            Keyword.STEPS_FROM_ALERT: spaces.Discrete(self.num_alert_prediction_steps + 2)
            # num_steps (1, 2, .. n) + no_alert (n + 1) + alert_time (0)
        })
        self.action_space = spaces.Discrete(2)  # wait , alert

        self.current_temp = None
        self.steps_from_alert = None
        self.P = self._create_transition()

        self.last_action = None
        self.last_delta = None

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.steps_from_alert == self.num_alert_prediction_steps + 1 and action == 0, \
            'Must choose wait action if alert is triggered'

        self.last_delta = np.random.choice(self.temperature_delta, size=1, p=self.prob)
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

                        for i in range(len(self.prob) - 1):
                            temp_delta = self.temperature_delta[i]

                            if temp_delta == -1:
                                prob = self.prob[i]
                            else:
                                prob = self.prob[i] + self.prob[i + 1]

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

                        for i in range(len(self.prob)):
                            self._append_to_transition(P,
                                                       prob=self.prob[i],
                                                       curr_temperature=curr_temp,
                                                       delta_temperature=self.temperature_delta[i],
                                                       curr_steps_from_alert=steps_from_alert,
                                                       delta_steps_from_alert=delta_step_from_alert,
                                                       action=action,
                                                       reward=reward,
                                                       done=False)

                    elif curr_temp == 1:
                        delta_step_from_alert = None

                        for i in range(len(self.prob)):
                            temp_delta = self.temperature_delta[i]
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
                                        reward = Reward.GOOD_ALERT
                                        delta_step_from_alert = 0

                            elif 1 < steps_from_alert < self.max_steps_from_alert:
                                if action == 0:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = Reward.GOOD_ALERT
                                else:
                                    continue

                            elif steps_from_alert == 1:
                                if action == 0:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = Reward.GOOD_ALERT
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
                                        reward = Reward.GOOD_ALERT
                                        delta_step_from_alert = self.max_steps_from_alert

                            self._append_to_transition(P,
                                                       prob=self.prob[i],
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
