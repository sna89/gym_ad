from gym.envs.registration import register
from constants import AD_V0_CONST, AD_V1_CONST

register(
    id='ad-v0',
    entry_point='gym_ad.envs:AdOneStepEnv',
    kwargs={'max_temp': AD_V0_CONST.MAX_TEMP, 'alert_prediction_steps': AD_V0_CONST.ALERT_PREDICTION_STEPS}
)

register(
    id='ad-v1',
    entry_point='gym_ad.envs:AdGaussianStepEnv',
    kwargs={'max_temp': AD_V1_CONST.MAX_TEMP, 'alert_prediction_steps': AD_V1_CONST.ALERT_PREDICTION_STEPS}

)
