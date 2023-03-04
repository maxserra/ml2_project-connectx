from typing import Dict
import numpy as np
import gym
from gym import spaces
from kaggle_environments import make


class ConnectX(gym.Env):

    def __init__(self,
                 env_configuration: Dict[str, int] = None,
                 opponent: str = "negamax",
                 switching_prob: float = 0.5,
                 debug: bool = False) -> None:

        # Set default arguments
        if env_configuration is None:
            env_configuration = {"rows": 6,
                                 "columns": 7,
                                 "inarow": 4}

        # Define custom fields
        self.env = make('connectx', configuration=env_configuration, debug=debug)
        self.pair = [None, opponent]
        self.trainer = self.env.train(self.pair)
        self.switching_prob = switching_prob

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = spaces.Discrete(config.columns)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(1, config.rows, config.columns),
                                            dtype=int)
        self.reward_range = (-10, 1)

    def step(self, action):
        # Check validity of action
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, info = self.trainer.step(int(action))
            reward = self._custom_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, info = -10, True, {}
        return self.obs, reward, done, info

    def reset(self):
        self._switch_trainer()
        self.obs = self.trainer.reset()
        return self.obs

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def _custom_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            config = self.env.configuration
            return 1 / (config.rows * config.columns)

    def _switch_trainer(self):
        if np.random.random() < self.switch_prob:
            self.pair = self.pair[::-1]  # reverse list order
            self.trainer = self.env.train(self.pair)
