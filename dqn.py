import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from blob import BlobEnv
import tqdm
from utils import ReplayBuffer

LEARNING_RATE = 0.001
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_SIZE = 1_000
MINIBTACH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5


class CNNDQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(CNNDQN, self).__init__()
        self.dqn = nn.Sequential(
            nn.Conv2d(num_inputs, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.Linear(64, actions_dim),
        )

    def forward(self, x):
        return self.dqn(x)


class CNNDQNAgent:
    def __init__(self, config):
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.model = CNNDQN(self.config.state_shape, self.config.action_dim)
        self.target_model = CNNDQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        if self.config.use_cuda:
            self.cuda()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()
