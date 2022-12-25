import numpy as np


def get_possible_actions(n_cols, board=None):
    if board is not None:
        return [str(x) for x in np.argwhere(np.array(board)[:n_cols] == 0).flatten()]
    else:
        return [str(x) for x in np.arange(n_cols)]


def convert_action_to_connectx_action(action):

    return int(action)
