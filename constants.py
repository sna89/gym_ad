
class AD_V0_CONST:
    MAX_TEMP = 5
    ALERT_PREDICTION_STEPS = 3
    PERIODS = 144 * 3


class AD_V1_CONST:
    MAX_TEMP = 40
    ALERT_PREDICTION_STEPS = 12
    PERIODS = 144 * 3


class Keyword:
    TEMPERATURE = 'temperature'
    STEPS_FROM_ALERT = 'steps_from_alert'


class Reward:
    MISSED_ALERT = -1000
    FALSE_ALERT = -950
    GOOD_ALERT = 10
