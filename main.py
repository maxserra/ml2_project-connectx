# import os
# import sys

from rl_agents.q_learning.q_learning import QLearningAgent
from utils import get_possible_actions, convert_action_to_connectx_action

# cwd = '/kaggle_simulations/agent/'
# if os.path.exists(cwd):
#     sys.path.append(cwd)
# else:
#     cwd = ''

# data = None


def agent_func(observation, configuration):

    actions = [str(x) for x in range(configuration["columns"])]
    q_agent = QLearningAgent(actions=actions)

    state = str(observation["board"])
    actions = get_possible_actions(n_cols=configuration["columns"], board=observation["board"])
    optimal_action = q_agent.get_optimal_action(state=state, actions=actions)

    return convert_action_to_connectx_action(action=optimal_action)
