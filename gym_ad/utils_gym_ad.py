from config import Keyword
import pandas as pd
from os import path
from config import Keyword
import plotly.express as px
from gym_ad.utils_taxi import decode_state


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


def render_ad(env, policy, gamma, steps=100):
    reward = 0
    state = env.reset()
    env.render()
    print()
    for t in range(steps):
        action = policy[state]
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


def add_to_report(policy, value_function, config):
    alert_prediction_steps = config["env"]["alert_prediction_steps"]
    reward_false_alert = config["reward"]["false_alert"]

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


def analyze_thts_statistics(statistics: pd.DataFrame()):
    # statistics = statistics.drop(columns=['Unnamed: 0'], axis=1)
    reward_stats = statistics.groupby("Run").sum()["Reward"]
    temperature_policy = {}
    for temperature in sorted(list(statistics["Temperature"].unique())):
        temperature_statistics_df = statistics[(statistics.Temperature == temperature) &
                                               (statistics.ignore == 0)]
        if temperature_statistics_df.shape[0] > 0:
            temperature_action_percentage = temperature_statistics_df[temperature_statistics_df.Action == 1].shape[0] \
                                            / float(temperature_statistics_df.shape[0])
            temperature_policy[temperature] = temperature_action_percentage
    temperature_policy_df = pd.DataFrame.from_dict(temperature_policy, orient='index')
    print(temperature_policy_df)
    print(reward_stats)


def add_step_to_statistic(env,
                          state,
                          statistics: pd.DataFrame(),
                          run: int,
                          action: int,
                          reward: float,
                          terminal: bool):
    data = dict()
    data['Run'] = run
    if env.spec.id == "ad-v1":
        data['state'] = "{}_{}".format(state[0], state[1])
    elif env.spec.id == "Taxi-v4":
        decoded_initial_state = decode_state(env, state)
        data['state'] = decoded_initial_state.__str__()

    data['Action'] = action
    data['Reward'] = reward
    data['Terminal'] = terminal
    if env.spec.id == "ad-v1":
        data['ignore'] = 1 if state[1] < env.max_steps_from_alert else 0

    step_statistic = pd.DataFrame(data=data, index=[0])
    statistics = pd.concat([statistics, step_statistic], axis=0, ignore_index=True)
    return statistics
