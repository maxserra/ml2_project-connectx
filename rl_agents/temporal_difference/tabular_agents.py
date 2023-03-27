import sys
import time
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, ReplayBufferSamples
from stable_baselines3.common.utils import is_vectorized_observation, get_linear_fn, safe_mean

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
        exploration_fraction: float = 0.25,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.005,
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

        self.exploration_rate = 0.0
        self.exploration_schedule = get_linear_fn(
            exploration_initial_eps,
            exploration_final_eps,
            exploration_fraction,
        )

        if _init_setup_model:
            super()._setup_model()

        self.policy: TabularQFunction

    def _on_step(self) -> None:
        """
        Update the exploration rate.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

        # Required due to how collect_rollouts works
        if self.env.envs[0].env.kaggle_env.done:
            self.env.envs[0].env.reset()

    def train(self, gradient_steps: int, batch_size: int = 1) -> None:

        q_value_gradients = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            state: str = str(replay_data.observations["board"].flatten().int().tolist())
            mark: int = replay_data.observations["mark"].flatten().int().tolist()[0]
            action: int = replay_data.actions.flatten().int().tolist()[0]
            next_state: str = str(replay_data.next_observations["board"].flatten().int().tolist())
            reward: float = replay_data.rewards.flatten().tolist()[0]

            q_old = self.policy.get_q_value(state=state, action=action)

            if not replay_data.dones:
                next_state_optimal_action = self.policy.get_optimal_action(state=next_state, player_mark=mark)
                q_next_state = self.policy.get_q_value(state=next_state, action=next_state_optimal_action)
            else:
                q_next_state = 0

            q_new = q_old + self.learning_rate * reward + self.gamma * q_next_state - q_old
            self.policy.set_q_value(state=state, action=action, value=q_new)

            q_value_gradients.append(abs(q_new - q_old))

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/q_value_gradient", safe_mean(q_value_gradients))

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
        selfplay_start: int = None,
        tb_log_name: str = "QLearning",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            # Enable selfplay by adding oneself to the opponents list
            if selfplay_start is not None and self._episode_num > selfplay_start:
                if hasattr(self.env.envs[0].env, "opponents"):
                    if self.kaggle_step_function not in self.env.envs[0].env.opponents:
                        print("adding selfplay")
                        self.env.envs[0].env.opponents.append(self.kaggle_step_function)
                else:
                    raise ValueError("For this selfplay approach to work, env should have an attribute 'opponents'.")

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", self._episode_num)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def kaggle_step_function(self, observation, configuration):

        print("playing against self!")

        state = str(observation["board"])
        mark = observation["mark"]

        optimal_action = self.policy.get_optimal_action(state=state, player_mark=mark)
        print(optimal_action)

        return optimal_action


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
        exploration_fraction: float = 0.50,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.005,
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

        q_value_gradients = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            state: str = str(replay_data.observations["board"].flatten().int().tolist())
            mark: int = replay_data.observations["mark"].flatten().int().tolist()[0]
            action: int = replay_data.actions.flatten().int().tolist()[0]
            next_state: str = str(replay_data.next_observations["board"].flatten().int().tolist())
            reward: float = replay_data.rewards.flatten().tolist()[0]

            q_old = self.policy.get_q_value(state=state, action=action)

            if not replay_data.dones:
                next_state_action = self.predict(observation=replay_data.next_observations)
                q_next_state = self.policy.get_q_value(state=next_state, action=next_state_action)
            else:
                q_next_state = 0

            q_new = q_old + self.learning_rate * reward + self.gamma * q_next_state - q_old
            self.policy.set_q_value(state=state, action=action, value=q_new)

            q_value_gradients.append(abs(q_new - q_old))

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/q_value_gradient", safe_mean(q_value_gradients))


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
        exploration_fraction: float = 0.50,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.005,
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

        q_value_gradients = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            state: str = str(replay_data.observations["board"].flatten().int().tolist())
            mark: int = replay_data.observations["mark"].flatten().int().tolist()[0]
            action: int = replay_data.actions.flatten().int().tolist()[0]
            next_state: str = str(replay_data.next_observations["board"].flatten().int().tolist())
            reward: float = replay_data.rewards.flatten().tolist()[0]

            q_old = self.policy.get_q_value(state=state, action=action)

            if not replay_data.dones:
                q_next_state = self.policy.get_state_expected_q_value(state=next_state)
            else:
                q_next_state = 0

            q_new = q_old + self.learning_rate * reward + self.gamma * q_next_state - q_old
            self.policy.set_q_value(state=state, action=action, value=q_new)

            q_value_gradients.append(abs(q_new - q_old))

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/q_value_gradient", safe_mean(q_value_gradients))


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
