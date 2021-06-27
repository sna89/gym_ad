from gym.envs.registration import register
from config import get_config


config_ad_v0 = get_config("AD_V0")
config_ad_v1 = get_config("AD_V1")

register(
    id='ad-v0',
    entry_point='gym_ad.envs:AdOneStepEnv',
    kwargs={"config": config_ad_v0}
)

register(
    id='ad-v1',
    entry_point='gym_ad.envs:AdCustomStepEnv',
    kwargs={"config": config_ad_v1}

)
