import gym
from gym_ad.envs.ad_env import AdEnv
from gym import spaces
from constants import Keyword, Reward
import math
import numpy as np


class AdGaussianStepEnv(AdEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_temp, alert_prediction_steps):
        super(AdGaussianStepEnv, self).__init__(max_temp, alert_prediction_steps)

    def _create_observation_space(self):
        observation_space = spaces.Dict({
            Keyword.TEMPERATURE: spaces.Discrete(self.max_temp + 1),
            # num_temp_levels + 0
            Keyword.STEPS_FROM_ALERT: spaces.Discrete(self.alert_prediction_steps + 1)
            # num_steps (1, 2, .. n) + no_alert (n + 1)
        })
        return observation_space

    def _create_action_space(self):
        return spaces.Discrete(2)

    def reset(self):
        self.current_state[Keyword.TEMPERATURE] = self.max_temp // 2
        self.current_state[Keyword.STEPS_FROM_ALERT] = self.max_steps_from_alert
        return self._get_obs()

    def _create_transition(self):
        P = self._init_transition_matrix()

        num_states_steps_from_alert = self.observation_space[Keyword.STEPS_FROM_ALERT].n
        num_states_temp = self.observation_space[Keyword.TEMPERATURE].n
        num_actions = self.action_space.n

        for curr_temp in range(num_states_temp):
            for steps_from_alert in range(1, num_states_steps_from_alert + 1):
                for action in range(num_actions):
                    if curr_temp == 0:
                        reward = 0
                        done = False
                        next_steps_from_alert = self.max_steps_from_alert - action
                        self._append_to_transition_wrapper(P,
                                                           prob=1,
                                                           curr_temperature=curr_temp,
                                                           next_temperature=self.max_temp // 2,
                                                           curr_steps_from_alert=steps_from_alert,
                                                           next_step_from_alert=next_steps_from_alert,
                                                           action=action,
                                                           reward=reward,
                                                           done=done)

                    else:
                        next_temp_possible_value_list, next_temp_prob_list = self._create_state_prob(curr_temp)
                        for next_temp, next_temp_prob in zip(next_temp_possible_value_list, next_temp_prob_list):
                            reward = 0
                            done = False
                            next_steps_from_alert = None

                            if next_temp == 0:
                                done = True
                                if steps_from_alert == self.max_steps_from_alert:
                                    if action == 0:
                                        reward = Reward.MISSED_ALERT
                                        next_steps_from_alert = steps_from_alert
                                    else:
                                        reward = self._get_good_alert_reward(steps_from_alert - 1)
                                        next_steps_from_alert = steps_from_alert - 1
                                elif 1 <= steps_from_alert < self.max_steps_from_alert:
                                    if action == 0:
                                        reward = self._get_good_alert_reward(steps_from_alert)
                                        if steps_from_alert == 1:
                                            next_steps_from_alert = self.max_steps_from_alert
                                        else:
                                            next_steps_from_alert = steps_from_alert - 1
                                    else:
                                        continue

                                self._append_to_transition_wrapper(P,
                                                                   prob=next_temp_prob,
                                                                   curr_temperature=curr_temp,
                                                                   next_temperature=next_temp,
                                                                   curr_steps_from_alert=steps_from_alert,
                                                                   next_step_from_alert=next_steps_from_alert,
                                                                   action=action,
                                                                   reward=reward,
                                                                   done=done)
                            elif next_temp > 0:
                                if steps_from_alert == self.max_steps_from_alert:
                                    if action == 0:
                                        next_steps_from_alert = steps_from_alert
                                    else:
                                        next_steps_from_alert = steps_from_alert - 1

                                elif 1 < steps_from_alert < self.max_steps_from_alert:
                                    if action == 0:
                                        next_steps_from_alert = steps_from_alert - 1
                                    else:
                                        continue

                                elif steps_from_alert == 1:
                                    reward = Reward.FALSE_ALERT
                                    if action == 0:
                                        next_steps_from_alert = self.max_steps_from_alert
                                    else:
                                        continue

                                self._append_to_transition_wrapper(P,
                                                                   prob=next_temp_prob,
                                                                   curr_temperature=curr_temp,
                                                                   next_temperature=next_temp,
                                                                   curr_steps_from_alert=steps_from_alert,
                                                                   next_step_from_alert=next_steps_from_alert,
                                                                   action=action,
                                                                   reward=reward,
                                                                   done=done)
        return P

    def _append_to_transition_wrapper(self,
                                      P,
                                      prob,
                                      curr_temperature,
                                      next_temperature,
                                      curr_steps_from_alert,
                                      next_step_from_alert,
                                      action,
                                      reward,
                                      done
                                      ):

        next_state = {Keyword.TEMPERATURE: next_temperature,
                      Keyword.STEPS_FROM_ALERT: next_step_from_alert}

        self._append_to_transition(P, curr_temperature, curr_steps_from_alert, action, prob,
                                   next_state, reward, done)

    def _create_state_prob(self, current_temp):
        score_func = lambda x: math.pow(2, -math.fabs(x - current_temp) / 5)
        next_temp_possible_value_list = list(current_temp + np.linspace(start=-10, stop=10, num=21))
        next_temp_score_list = [score_func(next_temp) for next_temp in next_temp_possible_value_list]
        score_sum = sum(next_temp_score_list)
        next_temp_prob_list = [next_temp_score / score_sum for next_temp_score in next_temp_score_list]

        if min(next_temp_possible_value_list) < 0:
            first_zero_occurrence = next_temp_possible_value_list.index(0)
            prob_zero_value = sum(next_temp_prob_list[:first_zero_occurrence + 1])
            del next_temp_prob_list[:first_zero_occurrence + 1]
            del next_temp_possible_value_list[:first_zero_occurrence]
            next_temp_prob_list.insert(0, prob_zero_value)

        if max(next_temp_possible_value_list) > self.max_temp:
            first_max_temp_occurrence = next_temp_possible_value_list.index(self.max_temp)
            prob_max_temp_value = sum(next_temp_prob_list[first_max_temp_occurrence:])
            del next_temp_prob_list[first_max_temp_occurrence:]
            del next_temp_possible_value_list[first_max_temp_occurrence + 1:]
            next_temp_prob_list.append(prob_max_temp_value)

        return next_temp_possible_value_list, next_temp_prob_list
