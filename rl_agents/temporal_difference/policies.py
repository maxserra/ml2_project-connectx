from typing import Dict, Optional, Tuple, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule

import utils


class TabularQFunction(BasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        q_values: Dict[str, Dict[str, float]] = None,
    ):
        super().__init__(
            observation_space,
            action_space
        )

        self.q_values: Dict[str, Dict[str, float]] = {} if q_values is None else q_values
        self.trained_on_mark = 1  # default to 1

    def get_q_value(self, state: str, action: str) -> float:
        '''Return stored q value for (state, action) pair or a random number if unknown.'''
        if isinstance(state, dict):
            state = self.features_extractor(state, self.trained_on_mark)
        if not isinstance(action, (str, int)):
            action = action.flatten().tolist()[0]
        if state not in self.q_values.keys():
            self.q_values[state] = {}
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = np.random.random()
        # return Q value
        return self.q_values[state][action]

    def get_state_expected_q_value(self, state: str) -> float:
        if state not in self.q_values.keys():
            return np.random.random()
        state_q_values = np.array(list(self.q_values[state].values()))
        return state_q_values.mean()

    def set_q_value(self, state: str, action: str, value: float):
        '''Store q value for (state, action) pair.'''
        if isinstance(state, dict):
            state = self.features_extractor(state, self.trained_on_mark)
        if not isinstance(action, (str, int)):
            action = action.flatten().tolist()[0]
        if not isinstance(value, float):
            value = value.flatten().tolist()[0]
        if state not in self.q_values.keys():
            self.q_values[state] = {}
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = 0
        # store value
        self.q_values[state][action] = value

    def get_optimal_action(self, state: str, player_mark: int):
        '''Return the action with highest q value for given state and action list.'''
        if state not in self.q_values.keys():
            return self.action_space.sample()

        optimal_action = self.action_space.sample()
        optimal_value = self.get_q_value(state=state, action=optimal_action)
        for action in self.q_values[state].keys():
            value = self.get_q_value(state=state, action=action)
            if player_mark == self.trained_on_mark:
                if value > optimal_value:
                    optimal_value = value
                    optimal_action = action
                elif value == optimal_value:
                    optimal_action = np.random.choice([optimal_action, action])
            else:
                if value < optimal_value:
                    optimal_value = value
                    optimal_action = action
                elif value == optimal_value:
                    optimal_action = np.random.choice([optimal_action, action])
        return optimal_action

    def forward(self, observation, deterministic: bool = True):
        return self._predict(observation={"state": str(observation["board"].flatten().tolist()),
                                          "mark": int(observation["mark"].flatten().tolist()[0])},
                             deterministic=deterministic)

    def _predict(self, observation, deterministic: bool = True):
        return self.get_optimal_action(state=observation["state"],
                                       player_mark=observation["mark"])

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        action = self._predict(observation={"state": str(observation["board"].flatten().tolist()),
                                            "mark": int(observation["mark"].flatten().tolist()[0])},
                               deterministic=deterministic)
        action = np.array([action]).reshape((-1,) + self.action_space.shape)
        return action, state

    def set_training_mode(self, mode: bool) -> None:
        return
