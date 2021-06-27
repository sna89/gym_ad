from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from config import Keyword
from gym_ad.envs.ad_env import AdEnv


class AdOneStepEnv(AdEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(AdOneStepEnv, self).__init__(config)

    def _create_observation_space(self):
        observation_space = spaces.Dict({
            Keyword.TEMPERATURE: spaces.Discrete(self.max_temp + 1),
            # num_temp_levels + temp_lower_threshold (0)
            Keyword.STEPS_FROM_ALERT: spaces.Discrete(self.alert_prediction_steps + 1)
            # num_steps (1, 2, .. n) + no_alert (n + 1)
        })
        return observation_space

    def _create_action_space(self):
        action_space = spaces.Discrete(2)  # wait , alert
        return action_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_state[Keyword.TEMPERATURE] = self.max_temp // 2 + 1
        self.current_state[Keyword.STEPS_FROM_ALERT] = self.max_steps_from_alert
        return self._get_obs()

    def close(self):
        pass

    def _get_obs(self):
        return self.current_state

    def _create_transition(self):
        # need to fix steps_from_alert == 0

        num_states_steps_from_alert = self.observation_space[Keyword.STEPS_FROM_ALERT].n
        num_states_temp = self.observation_space[Keyword.TEMPERATURE].n
        num_actions = self.action_space.n

        P = self._init_transition_matrix()

        prob_list = [float(1) / 3] * 3
        temperature_delta_list = [-1, 0, 1]

        for curr_temp in range(num_states_temp):
            for steps_from_alert in range(1, num_states_steps_from_alert):
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
                                reward = self.reward_false_alert
                            else:
                                continue

                        elif steps_from_alert == 0:
                            if action == 0:
                                delta_step_from_alert = self.max_steps_from_alert
                            else:
                                delta_step_from_alert = self.max_steps_from_alert - 1

                        for i in range(len(prob_list) - 1):
                            temp_delta = temperature_delta_list[i]

                            if temp_delta == -1:
                                prob = prob_list[i] + prob_list[i + 1]
                            else:
                                prob = prob_list[i]
                                temp_delta = 0

                            self._append_to_transition_wrapper(P,
                                                               prob=prob,
                                                               curr_temperature=curr_temp,
                                                               delta_temperature=temp_delta,
                                                               curr_steps_from_alert=steps_from_alert,
                                                               delta_steps_from_alert=delta_step_from_alert,
                                                               action=action,
                                                               reward=reward,
                                                               done=done)

                    elif 1 < curr_temp < self.max_temp:
                        reward = self.reward_false_alert if steps_from_alert == 1 else 0
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

                        for i in range(len(prob_list)):
                            self._append_to_transition_wrapper(P,
                                                               prob=prob_list[i],
                                                               curr_temperature=curr_temp,
                                                               delta_temperature=temperature_delta_list[i],
                                                               curr_steps_from_alert=steps_from_alert,
                                                               delta_steps_from_alert=delta_step_from_alert,
                                                               action=action,
                                                               reward=reward,
                                                               done=False)

                    elif curr_temp == 1:
                        delta_step_from_alert = None

                        for i in range(len(prob_list)):
                            temp_delta = temperature_delta_list[i]
                            done = True if temp_delta == -1 else False
                            reward = 0

                            if steps_from_alert == self.max_steps_from_alert:
                                if action == 0:
                                    delta_step_from_alert = 0
                                    if temp_delta == -1:
                                        reward = self.reward_missed_alert
                                else:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = self._get_good_alert_reward(steps_from_alert=self.alert_prediction_steps)

                            elif 1 < steps_from_alert < self.max_steps_from_alert:
                                if action == 0:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = self._get_good_alert_reward(steps_from_alert=steps_from_alert)
                                else:
                                    continue

                            elif steps_from_alert == 1:
                                if action == 0:
                                    delta_step_from_alert = -1
                                    if temp_delta == -1:
                                        reward = self._get_good_alert_reward(steps_from_alert=steps_from_alert)
                                    else:
                                        reward = self.reward_false_alert
                                else:
                                    continue

                            elif steps_from_alert == 0:
                                if action == 0:
                                    delta_step_from_alert = self.max_steps_from_alert
                                    if temp_delta == -1:
                                        reward = self.reward_missed_alert
                                else:
                                    delta_step_from_alert = self.max_steps_from_alert - 1
                                    if temp_delta == -1:
                                        reward = self._get_good_alert_reward(steps_from_alert=
                                                                             self.alert_prediction_steps)

                            self._append_to_transition_wrapper(P,
                                                               prob=prob_list[i],
                                                               curr_temperature=curr_temp,
                                                               delta_temperature=temp_delta,
                                                               curr_steps_from_alert=steps_from_alert,
                                                               delta_steps_from_alert=delta_step_from_alert,
                                                               action=action,
                                                               reward=reward,
                                                               done=done)

                    elif curr_temp == 0:
                        if action == 0:
                            temp_delta = (self.max_temp // 2)
                            reward = 0
                            done = False

                            delta_step_from_alert = self.max_steps_from_alert - steps_from_alert

                            self._append_to_transition_wrapper(P,
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

    def _append_to_transition_wrapper(self,
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

        self._append_to_transition(P, curr_temperature, curr_steps_from_alert, action, prob,
                                   next_state, reward, done)