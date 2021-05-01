import numpy as np
from typing import Dict, List
from constants import Constants, Keyword
from gym import spaces
import copy

"""
states = (current_temp, steps_from_alert) 
"""


def init_value_function(num_states_temp, num_states_steps):
    return {c_temp: {
        step: 0 for step in range(num_states_steps)
    } for c_temp in range(num_states_temp)}


def init_policy(num_states_temp, num_states_steps):
    return {c_temp: {
        step: 0 for step in range(num_states_steps)
    } for c_temp in range(num_states_temp)}


def init_actions(num_actions):
    return [i for i in range(num_actions)]


def init_states():
    return [(c_temp, steps) for c_temp in range(Constants.MAX_TEMP)
            for steps in range(Constants.ALERT_PREDICTION_STEPS + 2)]


def policy_iteration(P, state_space: spaces, action_space: spaces, gamma, tol: float = 10e-1):
    num_states_temp = state_space[Keyword.TEMPERATURE].n
    num_states_steps = state_space[Keyword.STEPS_FROM_ALERT].n

    value_function = None
    policy = init_policy(num_states_temp, num_states_steps)

    first_epoch = True
    new_policy = copy.deepcopy(policy)

    while first_epoch or not policy == new_policy:
        if first_epoch:
            first_epoch = False

        else:
            policy = new_policy.copy()
            del new_policy

        value_function = policy_evaluation(P, policy, state_space, gamma, tol)
        new_policy = policy_improvement(P, state_space, action_space, value_function, policy, gamma)

    policy = new_policy
    return value_function, policy


def policy_evaluation(P, policy, state_space, gamma, tol):
    num_states_temp = state_space[Keyword.TEMPERATURE].n
    num_states_steps = state_space[Keyword.STEPS_FROM_ALERT].n
    value_function = init_value_function(num_states_temp, num_states_steps)

    delta = np.inf
    while delta > tol:
        new_value_function = copy.deepcopy(value_function)
        for curr_temp in range(num_states_temp):
            for steps_from_alert in range(num_states_steps):
                action = policy[curr_temp][steps_from_alert]
                current_value = 0
                for prob, next_state, reward, done in P[curr_temp][steps_from_alert][action]:
                    current_value += prob * (reward +
                                             gamma *
                                             value_function[next_state[Keyword.TEMPERATURE]][next_state[Keyword.STEPS_FROM_ALERT]])
                new_value_function[curr_temp][steps_from_alert] = current_value

        delta = calc_delta_value_function(value_function, new_value_function)

        value_function = new_value_function
        del new_value_function

    return value_function


def policy_improvement(P, state_space, action_space, value_from_policy, policy, gamma):
    num_states_temp = state_space[Keyword.TEMPERATURE].n
    num_states_steps = state_space[Keyword.STEPS_FROM_ALERT].n
    new_policy = copy.deepcopy(policy)

    for curr_temp in range(num_states_temp):
        for steps_from_alert in range(num_states_steps):
            policy_action = policy[curr_temp][steps_from_alert]
            max_reward = value_from_policy[curr_temp][steps_from_alert]
            for alt_action in range(action_space.n):
                curr_reward = 0
                action_allowed = False
                for prob, next_state, reward, terminate in P[curr_temp][steps_from_alert][alt_action]:
                    action_allowed = True
                    curr_reward += prob * (reward +
                                           gamma *
                                           value_from_policy[next_state[Keyword.TEMPERATURE]][next_state[Keyword.STEPS_FROM_ALERT]])

                if curr_reward > max_reward and action_allowed:
                    max_reward = curr_reward
                    policy_action = alt_action

            new_policy[curr_temp][steps_from_alert] = policy_action

    return new_policy


def calc_delta_value_function(value_function, new_value_function):
    value_function_list = nested_dict_to_list(value_function)
    new_value_function_list = nested_dict_to_list(new_value_function)

    delta = np.max(np.abs(np.subtract(np.array(value_function_list),
                              np.array(new_value_function_list))))

    return delta


def nested_dict_to_list(nested_dict):
    new_l = list()
    for inner_dict in nested_dict.values():
        for value in inner_dict.values():
            new_l.append(value)
    return new_l
