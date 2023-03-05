from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, ReplayBufferSamples
from stable_baselines3.common.utils import is_vectorized_observation

from rl_agents.temporal_difference.policies import TabularQFunction


class QLearning(OffPolicyAlgorithm):

    def __init__(
        self,
        policy: Type[TabularQFunction],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.9,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 1,
        tau: float = 1.0,
        gamma: float = 1.0,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = -1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device="auto",
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_rate = exploration_final_eps

        if _init_setup_model:
            super()._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 1) -> None:
        # # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # # Update learning rate according to schedule
        # self._update_learning_rate(self.policy.optimizer)

        # losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            q_old = self.policy.get_q_value(state=replay_data.observations, action=replay_data.actions)

            if not replay_data.dones:
                new_optimal_action = self.policy.get_optimal_action(observation=replay_data.next_observations)
                q_new = self.policy.get_q_value(state=replay_data.next_observations, action=new_optimal_action)
            else:
                q_new = 0

            q_new = q_old + self.learning_rate * (replay_data.rewards + self.gamma * q_new - q_old)
            self.policy.set_q_value(state=replay_data.observations, action=replay_data.actions, value=q_new)

            # with th.no_grad():
            #     # Compute the next Q-values using the target network
            #     next_q_values = self.q_net_target(replay_data.next_observations)
            #     # Follow greedy policy: use the one with the highest value
            #     next_q_values, _ = next_q_values.max(dim=1)
            #     # Avoid potential broadcast issue
            #     next_q_values = next_q_values.reshape(-1, 1)
            #     # 1-step TD target
            #     target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # # Get current Q-values estimates
            # current_q_values = self.q_net(replay_data.observations)

            # # Retrieve the q-values for the actions from the replay buffer
            # current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # # Compute Huber loss (less sensitive to outliers)
            # loss = F.smooth_l1_loss(current_q_values, target_q_values)
            # losses.append(loss.item())

            # # Optimize the policy
            # self.policy.optimizer.zero_grad()
            # loss.backward()
            # # Clip gradient norm
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            # self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "QLearning",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


class Sarsa(QLearning):

    def __init__(
        self,
        policy: Type[TabularQFunction],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.9,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 1,
        tau: float = 1.0,
        gamma: float = 1.0,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = -1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super().__init__(
            self,
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            _init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 1) -> None:
        # # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # # Update learning rate according to schedule
        # self._update_learning_rate(self.policy.optimizer)

        # losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            q_old = self.policy.get_q_value(state=replay_data.observations, action=replay_data.actions)

            if not replay_data.dones:
                new_action = self.predict(observation=replay_data.next_observations)
                q_new = self.policy.get_q_value(state=replay_data.next_observations, action=new_action)
            else:
                q_new = 0

            q_new = q_old + self.learning_rate * (replay_data.rewards + self.gamma * q_new - q_old)
            self.policy.set_q_value(state=replay_data.observations, action=replay_data.actions, value=q_new)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/loss", np.mean(losses))


class ExpectedSarsa(QLearning):

    def __init__(
        self,
        policy: Type[TabularQFunction],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.9,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 1,
        tau: float = 1.0,
        gamma: float = 1.0,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = -1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super().__init__(
            self,
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            _init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 1) -> None:
        # # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # # Update learning rate according to schedule
        # self._update_learning_rate(self.policy.optimizer)

        # losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            q_old = self.policy.get_q_value(state=replay_data.observations, action=replay_data.actions)

            if not replay_data.dones:
                new_action = self.policy.get_state_expected_value(observation=replay_data.next_observations)
                q_new = self.policy.get_q_value(state=replay_data.next_observations, action=new_action)
            else:
                q_new = 0

            q_new = q_old + self.learning_rate * (replay_data.rewards + self.gamma * q_new - q_old)
            self.policy.set_q_value(state=replay_data.observations, action=replay_data.actions, value=q_new)

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/loss", np.mean(losses))


# class TabularQLearningAgent(TabularQFunctionAgentBase):

#     def __init__(self,
#                  actions: List[str],
#                  q_values: Dict[str, Dict[str, float]] = None,
#                  gamma: float = 1,  # reward discount factor
#                  alpha: float = 0.9,  # learning rate
#                  epsilon: float = 0.1,
#                  convergence_threshold: float = 1e-2) -> None:
#         '''Initialize with empty lookup table if none provided.'''
#         super().__init__(actions=actions, q_values=q_values, epsilon=epsilon, convergence_threshold=convergence_threshold)

#         # save hyperparameters
#         self.gamma = gamma
#         self.alpha = alpha

#     def update_q_value(self, old_state: str, action: str, new_state: str, reward: float, new_state_is_terminal: bool) -> float:
#         q_old = self.get_value(state=old_state, action=action)

#         if not new_state_is_terminal:
#             new_optimal_action = self.get_optimal_action(state=new_state)
#             q_new = self.get_value(state=new_state, action=new_optimal_action)
#         else:
#             q_new = 0

#         q_new = q_old + self.alpha * (reward + self.gamma * q_new - q_old)
#         self.set_value(state=old_state, action=action, value=q_new)
#         return q_new


# class TabularSarsaAgent(TabularQFunctionAgentBase):

#     def __init__(self,
#                  actions: List[str],
#                  q_values: Dict[str, Dict[str, float]] = None,
#                  gamma: float = 1,  # reward discount factor
#                  alpha: float = 0.9,  # learning rate
#                  epsilon: float = 0.1,
#                  convergence_threshold: float = 1e-2) -> None:
#         '''Initialize with empty lookup table if none provided.'''
#         super().__init__(actions=actions, q_values=q_values, epsilon=epsilon, convergence_threshold=convergence_threshold)

#         # save hyperparameters
#         self.gamma = gamma
#         self.alpha = alpha

#     def update_q_value(self, old_state: str, action: str, new_state: str, reward: float) -> float:
#         q_old = self.get_value(state=old_state, action=action)

#         new_optimal_action = self.get_epsilon_greedy_action(state=new_state)
#         q_next = self.get_value(state=new_state, action=new_optimal_action)

#         q_new = q_old + self.alpha * (reward + self.gamma * q_next - q_old)
#         self.set_value(state=old_state, action=action, value=q_new)
#         return q_new


# class TabularExpectedSarsaAgent(TabularQFunctionAgentBase):

#     def __init__(self,
#                  actions: List[str],
#                  q_values: Dict[str, Dict[str, float]] = None,
#                  gamma: float = 1,  # reward discount factor
#                  alpha: float = 0.9,  # learning rate
#                  epsilon: float = 0.1,
#                  convergence_threshold: float = 1e-2) -> None:
#         '''Initialize with empty lookup table if none provided.'''
#         super().__init__(actions=actions, q_values=q_values, epsilon=epsilon, convergence_threshold=convergence_threshold)

#         # save hyperparameters
#         self.gamma = gamma
#         self.alpha = alpha

#     def update_q_value(self, old_state: str, action: str, new_state: str, reward: float) -> float:
#         q_old = self.get_value(state=old_state, action=action)

#         new_optimal_action = self.get_state_expected_value(state=new_state)
#         q_next = self.get_value(state=new_state, action=new_optimal_action)

#         q_new = q_old + self.alpha * (reward + self.gamma * q_next - q_old)
#         self.set_value(state=old_state, action=action, value=q_new)
#         return q_new
