from gym_ad.meta_taxi import TaxiState, DestinationLocation, PassengerLocation
import numpy as np
from typing import Tuple
import time


def decode_state(env, encoded_state: int):
    taxi_state = [i for i in env.decode(encoded_state)]
    taxi_state[2] = PassengerLocation(taxi_state[2])
    taxi_state[3] = DestinationLocation(taxi_state[3])
    return TaxiState(*taxi_state)


def encode_state(env, state: TaxiState):
    enc_state = env.encode(state.taxi_row,
                           state.taxi_column,
                           state.passenger_loc.value,
                           state.destination.value)
    return enc_state


def taxi_env_step(env_enc_state, env, action, decode=True):
    env.s = env_enc_state
    state, reward, done, prob = env.step(action)
    if decode:
        state = decode_state(env, state)
    return state, reward, done, prob


def manhatten_dist(first_loc: Tuple, second_loc: Tuple):
    return np.abs(first_loc[0] - second_loc[0]) \
           + np.abs(first_loc[1] - second_loc[1])


def get_node_expandable_actions(env, node):
    expanded_actions = [successor.action for successor in node.successors]
    expandable_actions = [action for action in range(env.action_space.n) if action not in expanded_actions]
    return expandable_actions


def render_taxi(env, policy):
    done = False
    tot_reward = 0
    while not done:
        print(decode_state(env, env.s))
        env.render()
        current_state = env.s
        action = policy[current_state]
        state, reward, done, prob = taxi_env_step(current_state, env, action, decode=False)
        tot_reward += reward
        time.sleep(1)
    print(tot_reward)