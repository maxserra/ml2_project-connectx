import numpy as np
from agents.q_learning.q_learning import QLearningAgent


def agent_func(observation, configuration):

    actions = [str(x) for x in range(configuration["columns"])]
    q_agent = QLearningAgent(actions=actions)

    state = str(observation["board"])
    actions = [str(x) for x in np.argwhere(np.array(observation["board"])[:configuration["columns"]] == 0).flatten()]
    optimal_action = q_agent.get_optimal_action(state=state, actions=actions)

    return int(optimal_action)
