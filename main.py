import os
import sys

cwd = '/kaggle_simulations/agent/'
if os.path.exists(cwd):
    sys.path.append(cwd)
else:
    cwd = ''

agent = None


def agent_func(observation, configuration):
    global agent

    import utils

    if agent is None:
        agent = utils.load_pretrained_agent(file_name="TabularQLearningAgent_connectx_6x7row4.pkl",
                                            pretrained_agents_path="/kaggle_simulations/agent/pretrained_agents")

    return agent.agent_func(observation, configuration)
