import gym
import time
from gym_ad.algorithms import policy_iteration
from constants import Constants, Keyword
import numpy as np


def get_gamma():
    return 1 - 1 / float(Constants.PERIODS)


def render(env, policy, gamma, steps=100):
    reward = 0
    state = env.reset()
    env.render()
    for t in range(steps):
        action = policy[state[Keyword.TEMPERATURE]][state[Keyword.STEPS_FROM_ALERT]]
        state, step_reward, done, _ = env.step(action)
        reward = reward * gamma + step_reward
        print("Step Reward: {}".format(step_reward))
        env.render()
        time.sleep(1)

        if done:
            print("Current Reward: {}".format(reward))

    print("Final Reward: {}".format(reward))


if __name__ == "__main__":
    env = gym.make("gym_ad:ad-v0")
    gamma = get_gamma()
    value_function, policy = policy_iteration.policy_iteration(env.P, env.observation_space, env.action_space, gamma)
    render(env, policy, gamma, steps=Constants.PERIODS)