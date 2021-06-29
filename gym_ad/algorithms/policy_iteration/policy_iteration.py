import numpy as np
from config import Keyword
from gym import spaces
import copy
from utils import nested_dict_to_list

"""
states = (current_temp, steps_from_alert) 
"""


def get_gamma(periods):
    return 1 - 1 / float(periods)


def init_value_function(num_states_temp, num_states_steps):
    return {c_temp: {
        step: 0 for step in range(1, num_states_steps + 1)
    } for c_temp in range(num_states_temp)}


def init_policy(num_states_temp, num_states_steps):
    return {c_temp: {
        step: 0 for step in range(1, num_states_steps + 1)
    } for c_temp in range(num_states_temp)}


def policy_iteration(env, config):
    algorithm = "policy_iteration"
    gamma = get_gamma(config.get("algorithms").get(algorithm).get("periods"))
    tol = config.get("algorithms").get(algorithm).get("tol")

    P, state_space, action_space = env.P, env.observation_space, env.action_space
    num_states_temp = state_space[Keyword.TEMPERATURE].n
    num_states_steps = state_space[Keyword.STEPS_FROM_ALERT].n

    value_function = None
    first_epoch = True
    policy = init_policy(num_states_temp, num_states_steps)
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
            for steps_from_alert in range(1, num_states_steps + 1):
                action = policy[curr_temp][steps_from_alert]
                current_value = 0
                for prob, next_state, reward, done in P[curr_temp][steps_from_alert][action]:
                    current_value += prob * (reward +
                                             gamma *
                                             value_function[next_state.temperature][next_state.steps_from_alert])
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
        for steps_from_alert in range(1, num_states_steps + 1):
            policy_action = policy[curr_temp][steps_from_alert]
            max_reward = value_from_policy[curr_temp][steps_from_alert]
            for alt_action in range(action_space.n):
                curr_reward = 0
                action_allowed = False
                for prob, next_state, reward, terminate in P[curr_temp][steps_from_alert][alt_action]:
                    action_allowed = True
                    curr_reward += prob * (reward +
                                           gamma *
                                           value_from_policy[next_state.temperature][next_state.steps_from_alert])

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

