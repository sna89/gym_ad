from gym_ad.algorithms.max_uct.node import State, DecisionNode, ChanceNode
from copy import deepcopy
from utils import get_argmax_from_list, set_env_to_state
import pandas as pd


class MaxUCT:
    def __init__(self, env, config):
        self.simulator_env = env
        self.real_env = deepcopy(env)

        algorithm = "max_uct"
        self.num_trials = config.get("algorithms").get(algorithm).get("num_trials")
        self.trial_length = config.get("algorithms").get(algorithm).get("trial_length")
        self.num_actions = self.simulator_env.action_space.n
        self.uct_bias = config.get("algorithms").get(algorithm).get("uct_bias")
        self.runs = config.get("algorithms").get(algorithm).get("runs")

    def get_initial_state(self):
        initial_state = self.simulator_env.reset()
        return initial_state

    def run(self, state: State):
        statistics = pd.DataFrame()
        initial_node = DecisionNode(state, parent=None)

        for run in range(1, self.runs):
            node = deepcopy(initial_node)
            set_env_to_state(self.real_env, node.state)
            tot_reward = 0
            while True:
                for trial in range(self.num_trials):
                    self._run_trial(node)
                action = self.select_greedy_action(node)
                next_state, reward, terminal, _ = self.real_env.step(action)
                tot_reward += reward
                statistics = self.add_step_to_statistic(statistics, run, node.state, action, reward, terminal)

                if terminal:
                    break
                else:
                    node = DecisionNode(next_state, parent=node, terminal=False)
        return statistics

    def _run_trial(self, root_node: DecisionNode):
        depth = 0
        self._visit_decision_node(root_node, depth, False)

    def _visit_decision_node(self, decision_node: DecisionNode, depth: int, terminal: bool = False):
        if not terminal:
            decision_node.visit()
            if decision_node.is_first_visit():
                self._initialize_decision_node(decision_node)  # expansion
            chance_node = decision_node.select_chance_node()  # select action
            self._visit_chance_node(chance_node, depth)
        self._backup_decision_node(decision_node)

    def _visit_chance_node(self, chance_node: ChanceNode, depth: int):
        chance_node.visit()
        next_state, terminal = self._select_outcome(chance_node)
        if depth == self.trial_length - 1:
            terminal = True

        if terminal:
            decision_node = self.add_decision_node(next_state, chance_node, terminal=True)
        else:
            decision_node = self.add_decision_node(next_state, chance_node, terminal=False)

        self._visit_decision_node(decision_node, depth + 1, terminal)
        self._backup_chance_node(chance_node)

    def _select_outcome(self, chance_node: ChanceNode):
        set_env_to_state(self.simulator_env, chance_node.state)
        next_state, reward, terminal, _ = self.simulator_env.step(action=chance_node.action)  # monte carlo sample
        chance_node.reward = reward
        return next_state, terminal

    def _initialize_decision_node(self, decision_node: DecisionNode):
        steps_from_alert = decision_node.state.steps_from_alert
        if self.simulator_env.min_steps_from_alert <= steps_from_alert < self.simulator_env.max_steps_from_alert:
            self.add_chance_node(decision_node, action=0)
        else:
            for action in range(self.num_actions):
                self.add_chance_node(decision_node, action=action)

    @staticmethod
    def _backup_decision_node(decision_node: DecisionNode):
        decision_node.backup_max_uct()

    @staticmethod
    def _backup_chance_node(chance_node: ChanceNode):
        chance_node.backup_max_uct()

    @staticmethod
    def select_greedy_action(decision_node: DecisionNode):
        successor_values = [successor_node.value for successor_node in decision_node.successors]
        argmax_successor = get_argmax_from_list(successor_values, choose_random=True)
        greedy_action = decision_node.successors[argmax_successor].action
        return greedy_action

    def add_chance_node(self, decision_node: DecisionNode, action: int):
        chance_node = ChanceNode(state=decision_node.state,
                                 parent=decision_node,
                                 action=action,
                                 uct_bias=self.uct_bias)
        decision_node.add_successor(chance_node)

    @staticmethod
    def add_decision_node(next_state: State, chance_node: ChanceNode, terminal: bool = False):
        decision_node = DecisionNode(state=next_state, parent=chance_node, terminal=terminal)
        chance_node.add_successor(decision_node)
        return decision_node

    def add_step_to_statistic(self,
                              statistics: pd.DataFrame(),
                              run: int,
                              state: State,
                              action: int,
                              reward: float,
                              terminal: bool):
        data = dict()
        data['Run'] = run
        data['Temperature'] = state.temperature
        data['StepsFromAlert'] = state.steps_from_alert
        data['Action'] = action
        data['Reward'] = reward
        data['Terminal'] = terminal
        data['ignore'] = 1 if state.steps_from_alert < self.simulator_env.max_steps_from_alert else 0
        step_statistic = pd.DataFrame(data=data, index=[0])
        statistics = pd.concat([statistics, step_statistic], axis=0, ignore_index=True)
        return statistics
