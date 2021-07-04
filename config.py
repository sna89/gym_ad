import numpy as np


def get_config(env_name):
    config = {
        "env_name": env_name,
        "AD_V0": {
            "max_temp": 5,
            "alert_prediction_steps": 3,
            "reward": {
                "missed_alert": -1000,
                "false_alert": -100,
                "good_alert": 10,
            }
        },
        "AD_V1": {
            "max_temp": 40,
            "alert_prediction_steps": 12,
            "reward": {
                "missed_alert": -1000,
                "false_alert": -100,
                "good_alert": 10,
            }
        },
        "Taxi": {}
    }

    algorithms_config = {
        "PI": {
            "periods": 50,
            "tol": 1
        },
        "THTS": {
            "trial_length": 100,
            "num_trials": 2000,
            "runs": 1,
            "uct_bias": np.sqrt(2)
        }
    }

    config["algorithms"] = algorithms_config

    if env_name in ["AD_V0", "AD_V1", "Taxi"]:
        config["env"] = config[env_name]
    else:
        raise ValueError

    return config


class Keyword:
    TEMPERATURE = 'temperature'
    STEPS_FROM_ALERT = 'steps_from_alert'
