from config import Keyword
import pandas as pd
from os import path
from config import Keyword
import plotly.express as px


def plot_prob(policy):
    df = pd.DataFrame(columns=[Keyword.TEMPERATURE, 'Next Temperature', 'Probability'])
    for temp, steps in policy.items():
        if temp == 0:
            continue
        next_temp_prob_mapping = steps[1]
        base_dict = dict()
        base_dict[Keyword.TEMPERATURE] = temp
        for prob, next_state, _, _ in next_temp_prob_mapping[0]:
            next_temp = next_state[Keyword.TEMPERATURE]
            data = base_dict.copy()
            data['Next Temperature'] = next_temp
            data['Probability'] = prob
            df = df.append(data, ignore_index=True)
    fig = px.line(df, x="Next Temperature", y="Probability", color=Keyword.TEMPERATURE)
    fig.write_html('ProbabilityPlot.html')
    fig.show()

    # return df


def plot_value_function(csv_filename="report.csv", value_function=None):
    if path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        df = df.drop(columns=["Unnamed: 0"], axis=1)
    else:
        raise ValueError("File {} does not exists".format(csv_filename))

    fig = px.scatter(df, x="Value", y="False Alert Reward", color="Max Temp Alert",
                     size="Max Temp Alert",
                     hover_data=["Max Temp Alert", "Value"])
    fig.update_layout(title="False Alert reward vs Value", autosize=True)
    fig.show()
    fig.write_html('False Alert graph.html')


def update_reward(reward, gamma, step_reward):
    return reward * gamma + step_reward


def render(env, policy, gamma, steps=100):
    reward = 0
    state = env.reset()
    env.render()
    print()
    for t in range(steps):
        action = policy[state[Keyword.TEMPERATURE]][state[Keyword.STEPS_FROM_ALERT]]
        print("Action: {}".format(["Wait", "Alert"][action]))

        state, step_reward, done, _ = env.step(action)
        reward = update_reward(reward, gamma, step_reward)
        env.render()
        print("Step Reward: {}".format(step_reward))
        print()
        # time.sleep(1)

        if done:
            print("Current Reward: {}".format(reward))

    print("Final Reward: {}".format(reward))


def get_csv_report(report_name="report.csv", columns=("False Alert Reward", "Max Temp Alert", "Value")):
    if path.exists(report_name):
        df = pd.read_csv(report_name)
        df = df.drop(columns="Unnamed: 0", axis=1)
    else:
        df = pd.DataFrame(columns=columns)
    return df


def get_max_temp_alert_from_policy(policy, alert_prediction_steps):
    max_temp_alert = 0
    for temp, steps in policy.items():
        if steps[alert_prediction_steps + 1] == 1:
            max_temp_alert = temp
    return max_temp_alert


def add_to_report(policy, value_function, alert_prediction_steps, reward_false_alert):
    df = get_csv_report()

    max_temp_alert = get_max_temp_alert_from_policy(policy, alert_prediction_steps)
    value = value_function[max_temp_alert][alert_prediction_steps + 1]

    new_data = dict()
    new_data["False Alert Reward"] = reward_false_alert
    new_data["Max Temp Alert"] = max_temp_alert
    new_data["Value"] = value
    df = df.append(new_data, ignore_index=True)
    df.to_csv("report.csv")

    return max_temp_alert, value

