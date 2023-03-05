import numpy as np
import gym
from gym import spaces
from kaggle_environments import make


class TicTacToe(gym.Env):

    def __init__(self,
                 opponent: str = "reaction",
                 switching_prob: float = 0.5,
                 debug: bool = False) -> None:

        # Define custom fields
        self.env = make("tictactoe", debug=debug)
        self.pair = [None, opponent]
        self.trainer = self.env.train(self.pair)
        self.switching_prob = switching_prob

        # Define required gym fields (examples):
        self.action_space = spaces.Discrete(3 * 3)
        self.observation_space = spaces.Dict({'remainingOverageTime': spaces.Discrete(60 + 1),
                                              'step': spaces.Discrete(3 * 3 + 1),
                                              'board': spaces.Box(low=0, high=2, shape=(1, 3 * 3), dtype=int),
                                              'mark': spaces.Discrete(2 + 1)})
        self.reward_range = (-10, 1)

    def step(self, action):
        # Check validity of action
        is_valid = (self.obs['board'][int(action)] == 0) and self.action_space.contains(int(action))
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
        else:  # Reward 1/9
            return 1 / (3 * 3)

    def _switch_trainer(self):
        if np.random.random() < self.switching_prob:
            self.pair = self.pair[::-1]  # reverse list order
            self.trainer = self.env.train(self.pair)
