import numpy as np


def get_config(env_name):
    config = {
        "reward": {
            "missed_alert": -1000,
            "false_alert": -100,
            "good_alert": 10,
        }
    }

    env_config = {
        "AD_V0": {
            "max_temp": 5,
            "alert_prediction_steps": 3
        },
        "AD_V1": {
            "max_temp": 40,
            "alert_prediction_steps": 12
        }
    }

    algorithms_config = {
        "policy_iteration": {
            "periods": 24,
            "tol": 1
        },
        "max_uct": {
            "trial_length": 24,
            "num_trials": 500,
            "runs": 50,
            "uct_bias": np.sqrt(2)

        }
    }

    config["algorithms"] = algorithms_config

    if env_name in ["AD_V0", "AD_V1"]:
        config["env"] = env_config[env_name]
    else:
        raise ValueError

    return config


class Keyword:
    TEMPERATURE = 'temperature'
    STEPS_FROM_ALERT = 'steps_from_alert'
