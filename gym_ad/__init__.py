from gym.envs.registration import register

register(
    id='ad-v0',
    entry_point='gym_ad.envs:AdEnv',
)
