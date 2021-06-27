from gym_ad.algorithms.max_uct.node import State, DecisionNode


class MaxUCT:
    def __init__(self, env, config):
        self.env = env

        algorithm = "max_uct"
        self.num_trials = config.get("algorithms").get(algorithm).get("num_trials")
        self.trial_length = config.get("algorithms").get(algorithm).get("trial_length")

    def run(self, state: State):
        root_node = DecisionNode(state, parent=None, depth=0)

        for trial in range(self.num_trials):
            self.run_trial(root_node)

        action = self.select_action(root_node)
        return action

    def run_trial(self, root_node: DecisionNode):
        done = False
        depth = root_node.depth # depth = 0

        current_decision_node = root_node
        while not done and depth < self.trial_length:
            self.visit_decision_node(current_decision_node)


    def visit_decision_node(self, decision_node: DecisionNode):
        pass

    def select_action(self, root_node: DecisionNode):
        pass








