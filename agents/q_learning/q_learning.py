from typing import Dict, List
import numpy as np


class QLearningAgent:

    def __init__(self,
                 actions: List[str],
                 q_values: Dict[str, Dict[str, float]] = None,
                 gamma: float = 1,
                 lr: float = 1.0,
                 epsilon: float = 0.1,
                 threshold: float = 1e-2) -> None:
        '''Initialize with empty lookup table if none provided.'''
        self.actions = actions
        self.q_values: Dict[str, Dict[str, float]] = {} if q_values is None else q_values

        # save hyperparameters
        self.gamma = gamma
        self.learning_rate = lr
        self.threshold = threshold
        self.epsilon = epsilon  # exploration

        self.converged = False

    def get_value(self, state: str, action: str):
        '''Return stored q value for (state, action) pair or a random number if unknown.'''
        if state not in self.q_values.keys():
            self.q_values[state] = {}
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = abs(np.random.randn()) + 1
        # return value
        return self.q_values[state][action]

    def set_value(self, state: str, action: str, value: float):
        '''Store q value for (state, action) pair.'''
        if state not in self.q_values.keys():
            self.q_values[state] = {}
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = 0
        # store value
        self.q_values[state][action] = value

    def update_q_value(self, old_state: str, action: str, new_state: str, reward: float) -> float:
        q_old = self.get_value(state=old_state, action=action)

        new_optimal_action = self.get_optimal_action(state=new_state)
        q_next = self.get_value(state=new_state, action=new_optimal_action)

        q_new = q_old + self.learning_rate * (reward + self.gamma * q_next - q_old)
        self.set_value(state=old_state, action=action, value=q_new)
        return q_new

    def get_optimal_action(self, state: str, actions: List[str] = None, learning=True):
        '''Return the action with highest q value for given state and action list.'''
        actions = self.actions if actions is None else actions
        if not learning and state not in self.q_values.keys():
            return actions[0]

        max_value = - np.inf
        max_action = actions[0]
        for action in actions:
            if not learning and action not in self.q_values[state].keys():
                continue

            value = self.get_value(state=state, action=action)
            if value > max_value:
                max_value = value
                max_action = action
            elif value == max_value and learning:
                max_action = np.random.choice([max_action, action])
        return max_action

    def get_epsilon_greedy_action(self, state, actions, epsilon):
        '''Returns max_action or random action with probability of epsilon.'''
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        return self.get_optimal_action(state, actions)
