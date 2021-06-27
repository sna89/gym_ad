
def get_config(env_name):
    config = {
        "reward": {
            "missed_alert": -100,
            "false_alert": -10,
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
            "periods": 144 * 3,
            "tol": 1
        },
        "max_uct": {
            "trial_length": 18,
            "num_trials": 500
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
