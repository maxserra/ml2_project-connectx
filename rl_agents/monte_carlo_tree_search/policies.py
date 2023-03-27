from typing import Dict, Optional, Tuple, Union, Any, Callable, List, Type

import numpy as np
import scipy
import torch as th
from gym import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule

import utils


class MCTSPolicy(BasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        action_selection_strategy: str = "uct",
        use_sde: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=utils.custom_feature_extractor
        )

        action_selection_strategies = {
            "uct": self._compute_uct_values,
            "adaptive_uct": self._compute_adaptive_uct_values,
        }

        try:
            self.action_selection_function = action_selection_strategies[action_selection_strategy]
        except KeyError:
            raise ValueError(f"action_selection_strategy must be one of {action_selection_strategies.keys()}")

        self.trained_on_mark = 1  # default to 1

        self.tree = Tree(action_space=self.action_space)

    def get_optimal_action(self, state, player_mark):
        if not self.tree.contains(state=state, mark=self.trained_on_mark):
            # If the state doesn't exist in the tree, take random action
            return self.action_space.sample(), 0.0, 1.0

        state_node: TreeNode = self.tree.get(state=state, mark=self.trained_on_mark)
        values = self.action_selection_function(state_node.action_values,
                                                state_node.selection_count)
        if player_mark == self.trained_on_mark:
            action = max(values.keys(), key=lambda key: values[key])
            value = values[action]
        else:
            action = min(values.keys(), key=lambda key: values[key])
            value = -1 * values[action]
        return action, value, 1.0

    @staticmethod
    def _compute_uct_values(
        action_values: Dict[Any, float],
        selection_count: Dict[Any, int],
        exploration_cte: float = 2
    ) -> Dict[Any, float]:

        if sorted(action_values) != sorted(selection_count):
            raise RuntimeError("Unexpected difference in entries for 'action_values' and 'selections_count'." +
                               f"'{action_values.keys()}' and '{selection_count.keys()}'")

        actions_sorted = sorted(action_values.keys())
        values_sorted = np.array(list(map(action_values.get, actions_sorted)))
        count_sorted = np.array(list(map(selection_count.get, actions_sorted)))

        uct_values = values_sorted + exploration_cte * np.sqrt(count_sorted.sum()) / (1 + count_sorted)

        return {action: value for action, value in zip(actions_sorted, uct_values)}

    @staticmethod
    def _compute_adaptive_uct_values(
        action_values: Dict[Any, float],
        selection_count: Dict[Any, int]
    ) -> Dict[Any, float]:

        if sorted(action_values) != sorted(selection_count):
            raise RuntimeError("Unexpected difference in entries for 'action_values' and 'selections_count'." +
                               f"'{action_values.keys()}' and '{selection_count.keys()}'")

        actions_sorted = sorted(action_values.keys())
        values_sorted = np.array(list(map(action_values.get, actions_sorted)))
        count_sorted = np.array(list(map(selection_count.get, actions_sorted)))

        exploration_cte = scipy.special.softmax(values_sorted)

        uct_values = values_sorted + exploration_cte * np.sqrt(count_sorted.sum()) / (1 + count_sorted)

        return {action: value for action, value in zip(actions_sorted, uct_values)}

    def forward(
        self,
        observation,
        deterministic: bool = True
    ):
        return self._predict(observation={"state": str(observation["board"].flatten().tolist()),
                                          "mark": int(observation["mark"].flatten().tolist()[0])},
                             deterministic=deterministic)

    def _predict(self, observation, deterministic: bool = True):
        action, value, prob = self.get_optimal_action(state=observation["state"],
                                                      player_mark=observation["mark"])
        return (np.array([action]).reshape((-1,) + self.action_space.shape),
                th.tensor([value]),
                th.tensor([prob]))

    # Trick to make this policy behave like a nn.Model (needed for on_policy_alg.py:166)
    __call__: Callable[..., Any] = forward

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

    def predict_values(self, observation) -> th.Tensor:
        return th.tensor([self.forward(observation=observation)[1]])

    def set_training_mode(self, mode: bool) -> None:
        return


class TreeNode:

    def __init__(
        self,
        node_id,
        state,
        mark,
        action_space: spaces.Discrete,
        parent_id: str = None,
        is_terminal: bool = False,
        terminal_score: float = None
    ) -> None:

        self.node_id: str = node_id
        self.state: str = state
        self.mark: int = mark
        self.action_space = action_space
        self.parent_id: str = parent_id
        self.is_terminal: bool = is_terminal
        if is_terminal:
            self.children: Dict[Any, "TreeNode"] = None
            self.terminal_score: float = terminal_score
        else:
            self.children: Dict[Any, "TreeNode"] = {action: None for action in range(self.action_space.n)}
            self.terminal_score: float = None
        self.visits: int = 0

        self._value: float = 0.0

    def __eq__(self, __o: object) -> bool:
        return ((self.node_id == __o.node_id) and
                (self.state == __o.state) and
                (self.mark == __o.mark))

    def set_child(self, child, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Unexpected action '{action}' for action space '{self.action_space}'.")
        self.children[action] = child

    @property
    def action_values(self) -> Dict[Any, float]:
        return_dict = {}
        for action, child in self.children.items():
            if child is not None and child.node_id.endswith("_invalid"):
                continue
            return_dict[action] = child.value if child is not None else 0
        return return_dict
        # return {action: child.value for action, child in self.children.items() if child is not None}

    @property
    def selection_count(self) -> Dict[Any, int]:
        return_dict = {}
        for action, child in self.children.items():
            if child is not None and child.node_id.endswith("_invalid"):
                continue
            return_dict[action] = child.visits if child is not None else 0
        return return_dict
        # return {action: child.visits for action, child in self.children.items() if child is not None}

    @property
    def value(self):
        if self.is_terminal:
            return self.terminal_score
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class Tree:

    def __init__(
        self,
        action_space: spaces.Discrete
    ) -> None:

        self.action_space = action_space

        self.nodes_in_tree: Dict[str, TreeNode] = {}
        self.root = None

    @staticmethod
    def _generate_node_id(state, mark):
        return f"{state}"

    def _add_node(self, node_id: str, node: TreeNode):
        if node_id in self.nodes_in_tree:
            raise RuntimeError(f"node_id '{node_id}' is already in use. This is not supported.")
        self.nodes_in_tree[node_id] = node

    def contains(self, state, mark) -> bool:
        node_id = self._generate_node_id(state, mark)
        return node_id in self.nodes_in_tree.keys()

    def new_node(
        self,
        state,
        prev_state,
        prev_action,
        mark,
        node_id=None,
        is_terminal=False,
        terminal_score=None
    ) -> None:
        # print(state, prev_state, prev_action, mark)
        if prev_state is not None:
            parent_id = self._generate_node_id(prev_state, mark)
        else:
            parent_id = None
        node_id = self._generate_node_id(state, mark) if node_id is None else node_id
        node_obj = TreeNode(node_id=node_id,
                            state=state,
                            mark=mark,
                            action_space=self.action_space,
                            parent_id=parent_id,
                            is_terminal=is_terminal,
                            terminal_score=terminal_score,)

        self._add_node(node_id, node_obj)

        if parent_id is not None:
            # If there is a parent, add this node to the parent's children
            if parent_id not in self.nodes_in_tree:
                # If the parent is not yet present, make a new node
                self.new_node(state=None,
                              prev_state=None,
                              prev_action=None,
                              mark=mark,
                              node_id=parent_id)

            parent_obj = self.nodes_in_tree[parent_id]
            parent_obj.set_child(node_obj, prev_action)
        # elif self.root is None:
        #     # If there is no parent and no root, add the node as root
        #     self.root = node_obj
        # else:
        #     # If there is no parent but a root, make sure this node is the root
        #     if self.root != node_obj:
        #         raise RuntimeError(f"Mismatch in the root. Current root '{self.root}' is different than node '{node_obj}'.")

    def get(self, state, mark, node_id=None) -> Optional[TreeNode]:
        if node_id is None and self.contains(state, mark):
            return self.nodes_in_tree[self._generate_node_id(state, mark)]
        elif node_id is not None and node_id in self.nodes_in_tree.keys():
            return self.nodes_in_tree[node_id]
        return None
