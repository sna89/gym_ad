import gym

if __name__ == "__main__":
    env = gym.make("gym_ad:ad-v0")
    env.render()