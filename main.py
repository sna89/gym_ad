import gym
import time
from gym_ad.algorithms import policy_iteration
from constants import Constants
import numpy as np

if __name__ == "__main__":
    env = gym.make("gym_ad:ad-v0")
    policy = policy_iteration.policy_iteration(env.observation_space, env.action_space)
    env.render()
    for t in range(100):
        _, _, _, _ = env.step(0)
        env.render()
        time.sleep(3)