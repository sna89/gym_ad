

class Constants:
    MAX_TEMP = 10
    ALERT_PREDICTION_STEPS = 3
    PERIODS = 144


class Keyword:
    TEMPERATURE = 'temperature'
    STEPS_FROM_ALERT = 'steps_from_alert'


class Reward:
    MISSED_ALERT = -100
    FALSE_ALERT = -10
    GOOD_ALERT = 10