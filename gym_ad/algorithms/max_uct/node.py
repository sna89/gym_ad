from dataclasses import dataclass
# from typing import List


@dataclass
class State:
    temperature: int
    steps_from_alert: int
#    history: List


class Node:
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self._value = 0

        self.successors = []

    def add_successor(self, successor):
        self.successors.append(successor)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


class DecisionNode:
    def __init__(self, state: State, parent=None, depth=0):
        super(DecisionNode, self).__init__(state, parent)
        self._depth = depth

    @property
    def depth(self):
        return self._depth


class ChanceNode:
    def __init__(self, state: State, parent=None, action=None):
        super(ChanceNode, self).__init__(state, parent)
        self.action = action




