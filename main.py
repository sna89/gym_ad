import gym
from gym_ad.algorithms.max_uct.max_uct import MaxUCT
from config import get_config
import pandas as pd
from gym_ad.utils import analyze_max_uct_statistics
from gym_ad.algorithms.policy_iteration.policy_iteration import policy_iteration
from gym_ad.utils import add_to_report

if __name__ == "__main__":
    env = gym.make("gym_ad:ad-v1")

    env_name = "AD_V1"
    config = get_config(env_name)
    # max_uct_runner = MaxUCT(env, config)
    # initial_state = max_uct_runner.get_initial_state()
    # statistics = max_uct_runner.run(initial_state)
    #
    # statistics = pd.read_csv('statistics_12_steps.csv')
    # analyze_max_uct_statistics(statistics)
    value_function, policy = policy_iteration(env, config)
    print(policy)
    print(value_function)
    print(add_to_report(policy, value_function, config))
    # render(env, policy, gamma, steps=AD_V1_CONST.PERIODS)
    # df = plot_value_function()