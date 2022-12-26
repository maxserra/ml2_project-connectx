from typing import Dict, List
import numpy as np

import utils


class TabularQFunctionAgentBase:

    def __init__(self,
                 actions: List[str],
                 q_values: Dict[str, Dict[str, float]] = None,
                 epsilon: float = 0.1,
                 convergence_threshold: float = 1e-2) -> None:
        '''Initialize with empty lookup table if none provided.'''
        self.actions = actions
        self.q_values: Dict[str, Dict[str, float]] = {} if q_values is None else q_values
        self.trained_on_mark = 1  # default to 1

        # save hyperparameters
        self.epsilon = epsilon  # exploration
        self.convergence_threshold = convergence_threshold

        self.converged = False

    def get_value(self, state: str, action: str) -> float:
        '''Return stored q value for (state, action) pair or a random number if unknown.'''
        if state not in self.q_values.keys():
            self.q_values[state] = {}
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = np.random.random()
        # return Q value
        return self.q_values[state][action]

    def get_state_expected_value(self, state: str) -> float:
        if state not in self.q_values.keys():
            return np.random.random()
        state_q_values = np.array(list(self.q_values[state].values()))
        return state_q_values.mean()

    def set_value(self, state: str, action: str, value: float):
        '''Store q value for (state, action) pair.'''
        if state not in self.q_values.keys():
            self.q_values[state] = {}
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = 0
        # store value
        self.q_values[state][action] = value

    def update_q_value(self, **kwargs):
        raise NotImplementedError

    def get_optimal_action(self, state: str, actions: List[str] = None):
        '''Return the action with highest q value for given state and action list.'''
        actions = self.actions if actions is None else actions
        if state not in self.q_values.keys():
            return np.random.choice(actions)

        max_action = np.random.choice(actions)
        max_value = self.get_value(state=state, action=max_action)
        for action in actions:
            value = self.get_value(state=state, action=action)
            if value > max_value:
                max_value = value
                max_action = action
            elif value == max_value:
                max_action = np.random.choice([max_action, action])
        return max_action

    def get_epsilon_greedy_action(self, state: str, actions: List[str] = None, epsilon: float = None):
        '''Returns max_action or random action with probability of epsilon.'''
        actions = self.actions if actions is None else actions
        epsilon = self.epsilon if epsilon is None else epsilon

        if np.random.random() < epsilon:
            return np.random.choice(actions)
        return self.get_optimal_action(state=state, actions=actions)

    def agent_func(self, observation, configuration):

        state = str(observation["board"])
        mark = observation["mark"]

        if mark != self.trained_on_mark:
            state = utils.flip_state_marks(state=state, mark=mark, agent_trained_on_mark=self.trained_on_mark)

        actions = utils.get_possible_actions(n_cols=configuration["columns"], board=observation["board"])
        optimal_action = self.get_optimal_action(state=state, actions=actions)

        return utils.convert_action_to_connectx_action(action=optimal_action)
