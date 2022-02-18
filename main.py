import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from gym_game.envs.model import Agent

# import gym_game.envs.snake_game as g
from gym_game.envs.utils import plotLearning

from gym_game.envs.SnakeEnv import SnakeEnv

GAMMA = 0.99  # Weighting of future rewards
EPSILON = 1.0  # How long agents spends time exploring env vs taking best known action
LR = 0.003
BATCH_SIZE = 64
N_ACTIONS = 4
N_GAMES = 500
print("0")
if __name__ == "__main__":
    print("0.5")
    env = gym.make("Snake-v0")
    # f = SnakeEnv()
    print("1")
    score = 0
    print("2")
    agent = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
        n_actions=N_ACTIONS,
        eps_end=0.01,
        input_dims=3,
        lr=LR,
    )
    print("3")
    scores, eps_history = [], []
    print("4")
    for i in range(N_GAMES):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            print("action: ", action)
            observation_, reward, done, info = env.step(action)
            print("done 2")
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(
            "episode: ",
            i,
            " Score: %.2f " % score,
            " Avg score: %.2f" % avg_score,
            " Epsilon: %.2f" % agent.epsilon,
        )
    x = [i + 1 for i in range(N_GAMES)]
    filename = "snake_2022"
    plotLearning(x, scores, eps_history, filename)
