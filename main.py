import gym
from gym_ad.algorithms.policy_iteration import get_gamma, policy_iteration
from gym_ad.utils import add_to_report, render, get_csv_report, plot_value_function, plot_prob
from config import get_config


if __name__ == "__main__":
    env = gym.make("gym_ad:ad-v1")

    env_name = "AD_V1"
    config = get_config(env_name)



    # value_function, policy = policy_iteration(env, config)
    # print(policy)
    # print(value_function)
    # print(add_to_report(policy, value_function))
    # render(env, policy, gamma, steps=AD_V1_CONST.PERIODS)
    # df = plot_value_function()