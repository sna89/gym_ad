import numpy as np
from typing import Dict, List
from constants import Constants, Keyword
from gym import spaces

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


def get_gamma():
    return 1 - 1 / float(Constants.PERIODS)


def policy_iteration(states: spaces, actions: spaces, tol: float = 10e-3):
    num_states_temp = states[Keyword.TEMPERATURE].n
    num_states_steps = states[Keyword.STEPS_FROM_ALERT].n

    value_function = None
    policy = init_policy(num_states_temp, num_states_steps)
    gamma = get_gamma()

    first_epoch = True
    new_policy = policy.copy()

    while first_epoch or not policy == new_policy:
        if first_epoch:
            first_epoch = False

        else:
            policy = new_policy.copy()
            del new_policy

        value_function = policy_evaluation(policy, states, actions, gamma, tol)
        # new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)

    policy = new_policy
    return value_function, policy


def policy_evaluation(policy, states, actions, gamma, tol):
    num_states_temp = states[Keyword.TEMPERATURE].n
    num_states_steps = states[Keyword.STEPS_FROM_ALERT].n
    value_function = init_value_function(num_states_temp, num_states_steps)

    delta = np.inf
    while delta > tol:
        new_value_function = value_function.copy()
        for curr_temp in range(num_states_temp):
            for steps_from_alert in range(num_states_steps):
                state = (curr_temp, steps_from_alert)
                action = policy[curr_temp][steps_from_alert]
