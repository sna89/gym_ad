import numpy as np
from config import Keyword
from gym import spaces
import copy
from utils import nested_dict_to_list
import itertools

"""
states = (current_temp, steps_from_alert) 
"""


def get_gamma(periods):
    return 1 - 1 / float(periods)


def get_states(policy):
    for key in policy.keys():
        yield key


def init_value_function(env_id, state_space):
    return init_policy(env_id, state_space)


def init_policy(env_id, state_space):
    if env_id == "ad-v1":
        env_spaces = [k for k in state_space.spaces.values()]
        return {(c_temp, step): 0
                for step, c_temp in itertools.product(range(1, env_spaces[0].n + 1), range(env_spaces[1].n))
                }
    elif env_id == "Taxi-v4":
        return {state: 0 for state in range(state_space.n)}


def policy_iteration(env, config):
    algorithm = "PI"
    gamma = get_gamma(config.get("algorithms").get(algorithm).get("periods"))
    tol = config.get("algorithms").get(algorithm).get("tol")
    env_id = env.spec.id

    P, state_space, action_space = env.P, env.observation_space, env.action_space

    value_function = None
    first_epoch = True
    policy = init_policy(env_id, state_space)
    new_policy = copy.deepcopy(policy)

    while first_epoch or not policy == new_policy:
        if first_epoch:
            first_epoch = False

        else:
            policy = new_policy.copy()
            del new_policy

        value_function = policy_evaluation(env_id, P, policy, state_space, gamma, tol)
        new_policy = policy_improvement(P, action_space, value_function, policy, gamma)

    policy = new_policy
    return value_function, policy


def policy_evaluation(env_id, P, policy, state_space, gamma, tol):
    value_function = init_value_function(env_id, state_space)

    delta = np.inf
    while delta > tol:
        new_value_function = copy.deepcopy(value_function)
        for state in get_states(policy):
            action = policy[state]
            current_value = 0
            for prob, next_state, reward, done in P[state][action]:
                current_value += prob * (reward +
                                         gamma *
                                         value_function[next_state])
            new_value_function[state] = current_value

        delta = calc_delta_value_function(value_function, new_value_function)

        value_function = new_value_function
        del new_value_function

    return value_function


def policy_improvement(P, action_space, value_from_policy, policy, gamma):
    new_policy = copy.deepcopy(policy)

    for state in get_states(policy):
        policy_action = policy[state]
        max_reward = value_from_policy[state]
        for alt_action in range(action_space.n):
            curr_reward = 0
            action_allowed = False
            for prob, next_state, reward, terminate in P[state][alt_action]:
                action_allowed = True
                curr_reward += prob * (reward +
                                       gamma *
                                       value_from_policy[next_state])

            if curr_reward > max_reward and action_allowed:
                max_reward = curr_reward
                policy_action = alt_action

        new_policy[state] = policy_action

    return new_policy


def calc_delta_value_function(value_function, new_value_function):
    value_function_list = [val for val in value_function.values()]
    new_value_function_list = [val for val in new_value_function.values()]

    delta = np.max(np.abs(np.subtract(np.array(value_function_list),
                                      np.array(new_value_function_list))))

    return delta
