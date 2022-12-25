import os
import pickle
import numpy as np
from tqdm.autonotebook import tqdm

from kaggle_environments import Environment

from rl_agents.q_learning.q_learning import QLearningAgent
from utils import get_possible_actions, convert_action_to_connectx_action


def train_q_learning_agent(env: Environment,
                           starting_agent_path: str = None,
                           resulting_agent_path: str = None,
                           nr_episodes: int = 100, nr_steps: int = 100,
                           epsilon: float = 0.1, learning_rate: float = 0.9, gamma: float = 1):

    n_cols = env.configuration["columns"]

    # instanciate or load agent
    if starting_agent_path is None:
        agent = QLearningAgent(actions=get_possible_actions(n_cols=n_cols),
                               gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
    else:
        if os.path.exists(starting_agent_path):
            print(f"Loading pretrained agent from '{starting_agent_path}'...")
            with open(starting_agent_path, "rb") as file:
                agent = pickle.load(file)
        else:
            ValueError(f"Given pretrained agent path '{starting_agent_path}' not found.")

    for _ in tqdm(range(nr_episodes)):

        rival = np.random.choice(["random", "negamax"], size=1)[0]
        trainer = env.train([None, rival])

        observation = trainer.reset()

        for _ in range(nr_steps):

            board = observation["board"]

            state = str(board)
            actions = get_possible_actions(n_cols=n_cols, board=board)

            action = agent.get_optimal_action(state=state, actions=actions)

            observation, reward, done, info = trainer.step(convert_action_to_connectx_action(action))

            if reward is None:
                reward = -10
                print(done)

            next_state = str(observation["board"])

            agent.update_q_value(old_state=state, action=action, new_state=next_state, reward=reward)

            if done:
                print(f"Final state reached: \n  - {next_state}\n  - reward {reward}\n  - info {info}\n------")
                break  # final state reached

    if resulting_agent_path is None and starting_agent_path is None:
        print("No agent path provided. Trained agent won't be stored.")
        return agent
    elif resulting_agent_path is None and starting_agent_path is not None:
        resulting_agent_path = starting_agent_path

    print(f"Storing resulting trained agent to file '{resulting_agent_path}'...")
    with open(resulting_agent_path, 'wb') as file:
        pickle.dump(agent, file)

    return agent
