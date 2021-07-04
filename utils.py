from typing import List
import random


def normalize_list_values(l: List):
    score_sum = sum(l)
    normalized_list = list(map(lambda x: x / score_sum, l))
    return normalized_list


def nested_dict_to_list(nested_dict):
    new_l = list()
    for inner_dict in nested_dict.values():
        for value in inner_dict.values():
            new_l.append(value)
    return new_l


def get_argmax_from_list(l: List, choose_random=True):
    max_value = max(l)
    max_indices = [i for i, value in enumerate(l) if value == max_value]
    if choose_random:
        max_idx = random.choice(max_indices)
        return max_idx
    return max_indices


def set_env_to_state(env, state):
    if env.spec.id == "Taxi-v4":
        env.s = state
    elif env.spec.id == 'ad-v1':
        env.current_state = state

