import gym
from gym_ad.algorithms.thts.dp_uct import DpUCT
from config import get_config
from gym_ad.utils_gym_ad import analyze_thts_statistics
from gym_ad.algorithms.policy_iteration.policy_iteration import policy_iteration
from gym_ad.utils_taxi import decode_state, render_taxi
import os
import pandas as pd
from gym_ad.algorithms.policy_iteration.policy_iteration import get_gamma
from gym_ad.utils_gym_ad import render_ad


def get_env(env_name):
    if env_name == "Taxi":
        env = gym.make("gym_ad:Taxi-v4")
    elif env_name == "AD_V1":
        env = gym.make("gym_ad:ad-v1")
    else:
        raise ValueError
    return env


def run_algorithm(algorithm, env, config):
    if algorithm == "PI":
        value_function, policy = pi_main(env, config)
        return policy
    elif algorithm == "THTS":
        return thts_main(env, config)
    else:
        raise ValueError


def pi_main(env, config):
    value_function, policy = policy_iteration(env, config)
    return value_function, policy


def thts_main(env, config, render=True):
    dp_uct_runner = DpUCT(env, config)
    initial_state = dp_uct_runner.get_initial_state()
    statistics = dp_uct_runner.run(initial_state, render)
    return statistics


if __name__ == "__main__":
    env_name = os.getenv("ENV_NAME")
    algorithm = os.getenv("ALGORITHM")
    env = get_env(env_name)
    config = get_config(env_name)
    policy = run_algorithm(algorithm, env, config)
    if env_name == "Taxi" and algorithm == "PI":
        render_taxi(env, policy)
    if env_name == "AD_V1" and algorithm == "PI":
        gamma = get_gamma(config.get("algorithms").get(algorithm).get("periods"))
        render_ad(env, policy, gamma, steps=100)
