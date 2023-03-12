from typing import Any, Dict, Optional, Tuple, Type, Union
import warnings

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples, DictRolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, is_vectorized_observation
from stable_baselines3.common.vec_env import VecEnv

import utils

from rl_agents.monte_carlo_tree_search.policies import MCTSPolicy


class MCTS(OnPolicyAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[MCTSPolicy]],
        env: Union[GymEnv, str],
        max_episode_steps: int,
        learning_rate: Union[float, Schedule] = 0.9,
        n_steps: int = 1,
        gamma: float = 1,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,  # TODO
        vf_coef: float = 0.5,  # TODO
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device="auto",
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Discrete,),
        )

        if _init_setup_model:
            self._setup_model()

        # Override the rollout_buffer size to fit a whole episode
        self.rollout_buffer.buffer_size = max_episode_steps

        self.original_env_class = self.env.envs[0].env.__class__
        if self.env.num_envs > 1:
            warnings.warn("Current implementations assumes a single environment. " +
                          f"You have passed a '{type(self.env)}' with '{self.env.num_envs}'." +
                          "This might lead to unexpected behaviour.", NotImplementedError)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Adaptation of collect_rollouts to MCTS. This encompases the selection,
        expansion and simulation steps.

        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            raise NotImplementedError("No support for sde.")
            # self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                raise NotImplementedError("No support for sde.")
                # # Sample a new noise matrix
                # self.policy.reset_noise(env.num_envs)

            dones: np.ndarray = np.array([False] * env.num_envs)
            while not dones.all():
                # While not all envs are done

                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                if isinstance(actions, th.Tensor):
                    actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, spaces.Box):
                    raise NotImplementedError("No support for action space of type 'spaces.Box'.")
                    # clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped_actions)

                self.num_timesteps += env.num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos)
                n_steps += 1

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        print("These weird lines got hit...")
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[idx] += self.gamma * terminal_value

                if rollout_buffer.full:
                    raise RuntimeError("rollout_buffer is full. MCTS requires a complete episode, increase " +
                                       f"'max_episode_steps' accordingly. Current value '{rollout_buffer.size()}'")

                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
                self._last_obs = new_obs
                self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Adaptation of train to MCTS. This encompases the backup step.

        Update policy using the currently gathered rollout buffer.
        """
        # # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)

        self.policy: MCTSPolicy  # For helping putposes...

        # This will get all the data in the rollout_buffer one at a time in reversed order
        rollout_sample: Union[RolloutBufferSamples, DictRolloutBufferSamples]
        next_state: str = None
        for i, index in enumerate(reversed(range(self.rollout_buffer.size()))):
            rollout_sample = self.rollout_buffer._get_samples(batch_inds=index)

            state: str = str(rollout_sample.observations["board"].flatten().int().tolist())
            mark: int = rollout_sample.observations["mark"].flatten().int().tolist()[0]
            action: int = rollout_sample.actions.flatten().int().tolist()[0]
            reward: float = rollout_sample.returns.flatten().tolist()[0]

            from environments.tictactoe import TicTacToe
            self.original_env_class: TicTacToe  # For helping putposes...

            if i == 0:
                # The last entry in the buffer must have led to a terminal state. Notice the reversed order!
                # Figure out the intermediate state
                intermediate_state = self.original_env_class._determine_next_state(eval(state), action, mark)
                action_taken = self.original_env_class._determine_action_taken(eval(state), intermediate_state)
                if action_taken is None:
                    # If an invalid action was taken, intermediate_state led to terminal state
                    terminal_state_id = f"{intermediate_state}_{action}_invalid"
                    if terminal_state_id not in self.policy.tree.nodes_in_tree:
                        # If the intermediate state is not in the tree, add it
                        self.policy.tree.new_node(state=None,
                                                  prev_state=state,
                                                  prev_action=action,
                                                  mark=mark,
                                                  node_id=terminal_state_id,
                                                  is_terminal=True,
                                                  terminal_score=reward)

                    intermediate_state_node = self.policy.tree.get(state=None, mark=None, node_id=terminal_state_id)
                    intermediate_state_node.visits += 1
                else:
                    # If a legit action was taken, intermediate_state is a child of state
                    # Check if it was the final state
                    is_terminal = self.original_env_class._determine_if_terminal(intermediate_state)

                    if self.policy.tree._generate_node_id(intermediate_state, mark) not in self.policy.tree.nodes_in_tree:
                        # If the intermediate state is not in the tree, add it
                        self.policy.tree.new_node(state=intermediate_state,
                                                  prev_state=state,
                                                  prev_action=action,
                                                  mark=mark,
                                                  is_terminal=is_terminal,
                                                  terminal_score=reward if is_terminal else None)

                    intermediate_state_node = self.policy.tree.get(state=intermediate_state, mark=mark)
                    intermediate_state_node.visits += 1
                    intermediate_state_node.value = reward
            else:  # if next_state is not None:
                # All non-last entries are transitions between state and next_state
                # Figure out the intermediate state and which was the opponent action
                intermediate_state = self.original_env_class._determine_next_state(eval(state), action, mark)
                action_taken = self.original_env_class._determine_action_taken(eval(state), intermediate_state)
                if action_taken is None:
                    # If an invalid action was taken, intermediate_state led to terminal state
                    terminal_state_id = f"{intermediate_state}_{action}_invalid"
                    if terminal_state_id not in self.policy.tree.nodes_in_tree:
                        # If the intermediate state is not in the tree, add it
                        self.policy.tree.new_node(state=None,
                                                  prev_state=state,
                                                  prev_action=action,
                                                  mark=mark,
                                                  node_id=terminal_state_id,
                                                  is_terminal=True,
                                                  terminal_score=reward)

                    intermediate_state_node = self.policy.tree.get(state=None, mark=None, node_id=terminal_state_id)
                    intermediate_state_node.visits += 1
                else:
                    # If a legit action was taken, intermediate_state is a child of state
                    if self.policy.tree._generate_node_id(intermediate_state, mark) not in self.policy.tree.nodes_in_tree:
                        # If the intermediate state is not in the tree, add it
                        self.policy.tree.new_node(state=intermediate_state,
                                                  prev_state=state,
                                                  prev_action=action,
                                                  mark=mark)

                    # Get the action that the oppenent took
                    opponent_action = self.original_env_class._determine_action_taken(intermediate_state, eval(next_state))

                    # Set next_state as the child of the intermediate_state when taking opponent_action
                    intermediate_state_node = self.policy.tree.get(state=intermediate_state, mark=mark)
                    intermediate_state_node.set_child(child=self.policy.tree.get(state=next_state, mark=mark),
                                                      action=opponent_action)
                    intermediate_state_node.visits += 1
                    intermediate_state_node.value = np.array(list(intermediate_state_node.action_values.values())).mean()

            if self.policy.tree._generate_node_id(state, mark) not in self.policy.tree.nodes_in_tree:
                # If state is not in the tree, add it
                self.policy.tree.new_node(state=state,
                                          prev_state=None,
                                          prev_action=None,
                                          mark=mark)

            # Set intermediate_state as the child of state when taking action
            state_node = self.policy.tree.get(state=state, mark=mark)
            state_node.set_child(child=intermediate_state_node,
                                 action=action)
            # Update state  with the information from the (freshly updated) child
            state_node.visits += 1
            state_node.value = np.array(list(state_node.action_values.values())).mean()

            # Store state as next_state for next iteration
            next_state = state

        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1

        # Logs
        self.logger.record("train/n_updates", self._n_updates)#, exclude="tensorboard")
        # self.logger.record("train/explained_variance", explained_var)
        # self.logger.record("train/entropy_loss", entropy_loss.item())
        # self.logger.record("train/policy_loss", policy_loss.item())
        # self.logger.record("train/value_loss", value_loss.item())
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MCTS",
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

    def kaggle_step_function(self, observation, configuration):

        state = str(observation["board"])
        mark = observation["mark"]

        # if mark != self.policy.trained_on_mark:
        #     state = utils.flip_state_marks(state=state, mark=mark, agent_trained_on_mark=self.policy.trained_on_mark)

        # actions = utils.get_possible_actions(n_cols=configuration["columns"], board=observation["board"])
        optimal_action = self.policy.get_optimal_action(state=state, player_mark=mark)[0]
        print(optimal_action)

        # return utils.convert_action_to_connectx_action(action=optimal_action)
        return optimal_action
