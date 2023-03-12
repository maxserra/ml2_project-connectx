import os
import lzma
import pickle
from typing import Dict, List
import numpy as np


def get_possible_actions(n_cols, board=None):
    if board is not None:
        return [str(x) for x in np.argwhere(np.array(board)[:n_cols] == 0).flatten()]
    else:
        return [str(x) for x in np.arange(n_cols)]


def convert_action_to_connectx_action(action):

    return int(action)


def generate_agent_file_name(agent, env):

    agent_class = type(agent).__name__
    env_name = env.name
    env_rows = env.configuration["rows"]
    env_cols = env.configuration["columns"]
    env_inarow = env.configuration["inarow"]

    return f"{agent_class}_{env_name}_{env_rows}x{env_cols}row{env_inarow}.pkl"


def flip_state_marks(state: str, mark: int, agent_trained_on_mark: int):

    if mark == agent_trained_on_mark:
        return state
    else:
        new_state = state.replace("1", "temp")
        new_state = new_state.replace("2", "1")
        new_state = new_state.replace("temp", "2")
        return new_state


def load_pretrained_agent(file_name: str,
                          pretrained_agents_path: str = "pretrained_agents"):

    file_path = os.path.join(pretrained_agents_path, file_name + ".pkl.xz")
    if os.path.exists(file_path):
        print(f"Loading pretrained agent from '{file_path}'...")
        with lzma.open(file_path, "rb") as file:
            agent = pickle.load(file)
    else:
        raise ValueError(f"Given pretrained agent path '{file_path}' not found.")

    return agent


def dump_pretrained_agent(agent,
                          file_name: str,
                          pretrained_agents_path: str = "pretrained_agents",
                          overwrite: bool = True) -> bool:

    file_path = os.path.join(pretrained_agents_path, file_name + ".pkl.xz")
    if os.path.exists(file_path) and not overwrite:
        print(f"Provided file already exists '{file_path}' and it will NOT be overwritten.")
        return False

    print(f"Dumping pretrained agent to '{file_path}'...")
    with lzma.open(file_path, "wb") as file:
        pickle.dump(agent, file)

    return True


def custom_feature_extractor(observation: Dict, trained_on: int) -> str:
    if observation is None:
        return None
    if isinstance(observation, List):
        return [custom_feature_extractor(obs, trained_on) for obs in observation]

    state = flip_state_marks(state=str(observation["board"].flatten().tolist()),
                             mark=int(observation["mark"].flatten().tolist()[0]),
                             agent_trained_on_mark=trained_on)
    return state
