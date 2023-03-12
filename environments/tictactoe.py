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
        self.kaggle_env = make("tictactoe", debug=debug)
        self.pair = [None, opponent]
        self.trainer = self.kaggle_env.train(self.pair)
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
        return self.kaggle_env.render(**kwargs)

    def run(self, agents):
        temp_env = self.kaggle_env.clone()
        temp_env.reset()
        temp_env.run(agents)

    def play(self, agents=None, **kwargs):
        temp_env = self.kaggle_env.clone()
        temp_env.reset()
        temp_env.play(agents, **kwargs)

    @staticmethod
    def _custom_reward(old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/9
            return 1 / (3 * 3)

    @staticmethod
    def _determine_next_state(origin_state, action, mark):
        """Get the state that happened when 'action' was taken in 'origin_state'.

        This function is intended for postprocessing. NOT FOR PLAYING."""
        if not (isinstance(origin_state, list) and
                isinstance(action, int) and
                isinstance(mark, int)):
            raise TypeError(f"Invalid types: {type(origin_state), type(action), type(mark)}")

        next_state = origin_state.copy()
        if next_state[action] == 0:
            next_state[action] = mark
        return next_state

    @staticmethod
    def _determine_action_taken(origin_state, next_state):
        """Get the action that was played in 'origin_state' to end up in 'next_state'.

        This function is intended for postprocessing. NOT FOR PLAYING."""
        if not (isinstance(origin_state, list) and
                isinstance(next_state, list)):
            raise TypeError(f"Invalid types: {type(origin_state), type(next_state)}")

        action_taken = np.argwhere(np.array(next_state) - np.array(origin_state)).flatten()
        if len(action_taken) == 0:
            return None
        if len(action_taken) > 1:
            raise ValueError("This function is only ment for ONE STEP transitions, " +
                             f"but two changes were detected: '{action_taken.tolist()}'.")
        return action_taken[0]

    @staticmethod
    def _determine_if_terminal(state):
        """Get whether 'state' is terminal.

        This function is intended for postprocessing. NOT FOR PLAYING."""
        if not isinstance(state, list):
            raise TypeError(f"Invalid types: {type(state)}")

        return all(state)

    def _switch_trainer(self):
        if np.random.random() < self.switching_prob:
            self.pair = self.pair[::-1]  # reverse list order
            self.trainer = self.kaggle_env.train(self.pair)
