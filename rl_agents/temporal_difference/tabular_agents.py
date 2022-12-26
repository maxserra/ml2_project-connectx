from typing import Dict, List

from rl_agents.base_agents import TabularQFunctionAgentBase


class TabularQLearningAgent(TabularQFunctionAgentBase):

    def __init__(self,
                 actions: List[str],
                 q_values: Dict[str, Dict[str, float]] = None,
                 gamma: float = 1,  # reward discount factor
                 alpha: float = 0.9,  # learning rate
                 epsilon: float = 0.1,
                 convergence_threshold: float = 1e-2) -> None:
        '''Initialize with empty lookup table if none provided.'''
        super().__init__(actions=actions, q_values=q_values, epsilon=epsilon, convergence_threshold=convergence_threshold)

        # save hyperparameters
        self.gamma = gamma
        self.alpha = alpha

    def update_q_value(self, old_state: str, action: str, new_state: str, reward: float, new_state_is_terminal: bool) -> float:
        q_old = self.get_value(state=old_state, action=action)

        if not new_state_is_terminal:
            new_optimal_action = self.get_optimal_action(state=new_state)
            q_new = self.get_value(state=new_state, action=new_optimal_action)
        else:
            q_new = 0

        q_new = q_old + self.alpha * (reward + self.gamma * q_new - q_old)
        self.set_value(state=old_state, action=action, value=q_new)
        return q_new


class TabularSarsaAgent(TabularQFunctionAgentBase):

    def __init__(self,
                 actions: List[str],
                 q_values: Dict[str, Dict[str, float]] = None,
                 gamma: float = 1,  # reward discount factor
                 alpha: float = 0.9,  # learning rate
                 epsilon: float = 0.1,
                 convergence_threshold: float = 1e-2) -> None:
        '''Initialize with empty lookup table if none provided.'''
        super().__init__(actions=actions, q_values=q_values, epsilon=epsilon, convergence_threshold=convergence_threshold)

        # save hyperparameters
        self.gamma = gamma
        self.alpha = alpha

    def update_q_value(self, old_state: str, action: str, new_state: str, reward: float) -> float:
        q_old = self.get_value(state=old_state, action=action)

        new_optimal_action = self.get_epsilon_greedy_action(state=new_state)
        q_next = self.get_value(state=new_state, action=new_optimal_action)

        q_new = q_old + self.alpha * (reward + self.gamma * q_next - q_old)
        self.set_value(state=old_state, action=action, value=q_new)
        return q_new


class TabularExpectedSarsaAgent(TabularQFunctionAgentBase):

    def __init__(self,
                 actions: List[str],
                 q_values: Dict[str, Dict[str, float]] = None,
                 gamma: float = 1,  # reward discount factor
                 alpha: float = 0.9,  # learning rate
                 epsilon: float = 0.1,
                 convergence_threshold: float = 1e-2) -> None:
        '''Initialize with empty lookup table if none provided.'''
        super().__init__(actions=actions, q_values=q_values, epsilon=epsilon, convergence_threshold=convergence_threshold)

        # save hyperparameters
        self.gamma = gamma
        self.alpha = alpha

    def update_q_value(self, old_state: str, action: str, new_state: str, reward: float) -> float:
        q_old = self.get_value(state=old_state, action=action)

        new_optimal_action = self.get_state_expected_value(state=new_state)
        q_next = self.get_value(state=new_state, action=new_optimal_action)

        q_new = q_old + self.alpha * (reward + self.gamma * q_next - q_old)
        self.set_value(state=old_state, action=action, value=q_new)
        return q_new
