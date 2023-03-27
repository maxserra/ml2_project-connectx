import argparse
from pprint import pprint
from typing import Optional

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback

from rl_agents.monte_carlo_tree_search.mcts_agent import MCTS, MCTSPolicy
from rl_agents.temporal_difference.tabular_agents import QLearning, Sarsa, ExpectedSarsa, TabularQFunction
from environments.tictactoe import TicTacToe


def main_MCTS(project_name, config):

    run = wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    def make_env():
        env = TicTacToe(opponents=config["env_config"]["opponents"],
                        switching_prob=config["env_config"]["switching_prob"])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])

    mcts_agent = MCTS(policy=MCTSPolicy,
                      env=env,
                      max_episode_steps=10,
                      verbose=1,
                      tensorboard_log=f"logs/{config['algorithm']}_tictactoe"
                      )

    mcts_agent.learn(total_timesteps=config["total_timesteps"],
                     log_interval=config["log_interval"],
                     selfplay_start=config["selfplay_start"],
                     reset_num_timesteps=False,
                     callback=WandbCallback(log="all",
                                            model_save_path=f"pretrained_agents/{run.name}",
                                            )
                     )


def main_QLearning(project_name, config):

    run = wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    def make_env():
        env = TicTacToe(opponents=config["env_config"]["opponents"],
                        switching_prob=config["env_config"]["switching_prob"])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])

    agent = eval(config["algorithm"])(policy=TabularQFunction,
                                      env=env,
                                      learning_starts=0,
                                      verbose=1,
                                      tensorboard_log=f"logs/{config['algorithm']}_tictactoe"
                                      )

    agent.learn(total_timesteps=config["total_timesteps"],
                log_interval=config["log_interval"],
                selfplay_start=config["selfplay_start"],
                reset_num_timesteps=False,
                callback=WandbCallback(log="all",
                                       model_save_path=f"pretrained_agents/{run.name}",
                                       )
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("project_name")
    parser.add_argument("-algo", required=True)
    parser.add_argument("-policy", required=True)
    parser.add_argument("-env", required=True)
    parser.add_argument("-env_opponents", nargs="+", required=True)
    parser.add_argument("-env_switching", required=True, type=float)
    parser.add_argument("-timesteps", required=True, type=int)
    parser.add_argument("-log_inter", required=True, type=int)
    parser.add_argument("-selfplay", required=False, type=int, default=None)

    args = parser.parse_args()

    project_name = args.project_name

    config = {
        "algorithm": args.algo,
        "policy": args.policy,
        "env": args.env,
        "env_config": {"opponents": args.env_opponents,
                       "switching_prob": args.env_switching},
        "total_timesteps": args.timesteps,
        "log_interval": args.log_inter,
        "selfplay_start": args.selfplay,
    }

    # python arena.py project_a -algo MCTS -policy MCTSPolicy -env TicTacToe -env_opponents random reaction -env_switching 0.5 -timesteps 10000 -log_inter 100 -selfplay 1000

    # config = {
    #     "algorithm": "MCTS",
    #     "policy": "MCTSPolicy",
    #     "env": "TicTacToe",
    #     "env_config": {"opponents": ["random", "reaction"],
    #                    "switching_prob": 0.5},
    #     "total_timesteps": 10000,
    #     "log_interval": 100,
    #     "selfplay_start": 1000
    # }

    if config["algorithm"] == "MCTS":
        main_MCTS(project_name=project_name, config=config)
    elif config["algorithm"] in ["QLearning", "Sarsa", "ExpectedSarsa"]:
        main_QLearning(project_name=project_name, config=config)
