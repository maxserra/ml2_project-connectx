# ml2_project-connectx

ML2 project for WS22/23.

RL algorithms and the ConnectX Kaggle competition.

## Focus of the project

Two player fully-observable games (e.g. tic-tac-toe, connect4, backgammon, chess) are learnable. In this project I want to explore a range of algorithms and compare them. Ultimately, I want to have a hih ranking submission in the ConnectX Kaggle competition.

Algorithms to try:
- Non-deep:
    - Q-learning (and it's TD variations): Learn a explicit Q function for state-action pairs
    - Monte Carlo tree search
- Deep:
    - Deep Q learning: Train a NN to calculate the Q value for state-action pairs
    - Policy Gradient: Train a NN to determine the optimal action for each state
    - Proximal Policy Optimization
- Genetic algorithms:
    - Neuroevolution: Create a population of candidate (deep) agents with different weights (and potentially different NN architecture). This population will evolve to a new generation by selecting the fittest and applying mutations and crossover to their "DNA" (weights and NN params).

### General considerations

How to evaluate the agents:
- The "intuitive" metric of win rate against a fixed agent is a special case of the more general average cumulative reward when the reward is a simple 1-0.
- Using average cumulative reward as an evaluation metric will allow us to use a (slightly) more elaborate reward (see discussion of invalid actions below).


How to train the agents:
- No access to the rules of the games is given, and therefore the agent will have to learn them. This translates in that the agent will be allowed to select any action, even the invalid ones, and when an invalid action is selected the reward will be (very) negative.
- Learn from an "expert": Play against a given agent (or multiple ones).
    - This, however, will not enable our agent to become substantially better than the "teachers". 
    - Evaluation: "time to beat the teacher" and "how much better than teacher can one be".
- Learn by self-play: Play against previous versions of one-self. No access to an expert allowed during training.
    - Although not used during training, here an "expert" will still be needed to provide a baseline.
    - Evaluation: "time to beat the expert" "how much better than teacher can one be"


How do define the reward function:
- Vanilla (and the least interventionist one): Win 1, Lose 0
- Connect4 specific:
    - Points for 2-in-a-row and 3-in-a-row connections.
    - Points for less steps to victory, more steps to defeat.
