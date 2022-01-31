import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Agent
import snake_game as g
from utils import plotLearning
from SnakeEnv import SnakeEnv

GAMMA = 0.99  # Weighting of future rewards
EPSILON = 1.0  # How long agents spends time exploring env vs taking best known action
LR = 0.003
BATCH_SIZE = 64
N_ACTIONS = 4

# if __name__ == "__main__":
#     env = g.Game()
#     agent = Agent(
#         gamma=GAMMA,
#         epsilon=EPSILON,
#         batch_size=BATCH_SIZE,
#         n_actions=N_ACTIONS,
#         eps_end=0.01,
#         input_dims=[4],
#         lr=LR,
#     )
#     scores, eps_history = [], []
