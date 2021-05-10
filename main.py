import gym
from gym_ad.algorithms.policy_iteration import get_gamma, policy_iteration
from gym_ad.utils import add_to_report, render, get_csv_report, plot_value_function, plot_prob


if __name__ == "__main__":
    env = gym.make("gym_ad:ad-v1")
    plot_prob(env.P)
    # gamma = get_gamma()

    # value_function, policy = policy_iteration(env.P, env.observation_space, env.action_space, gamma)
    # print(policy)
    # print(value_function)
    # print(add_to_report(policy, value_function))
    # render(env, policy, gamma, steps=AD_V1_CONST.PERIODS)
    # df = plot_value_function()