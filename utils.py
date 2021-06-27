from typing import List


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