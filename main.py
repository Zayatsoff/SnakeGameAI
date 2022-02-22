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
from dqn import CNNDQNAgent
import gym


# dictionairy
config = {
    "env": "CartPole-v0",
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "eps_decay": 30000,
    "episodes": 2000000,
    "use_cuda": True,
    "learning_rate": 1e-4,
    "max_buff": 100000,
    "update_tar_interval": 1000,
    "batch_size": 32,
    "print_interval": 5000,
    "checkpoint": True,
    "checkpoint_interval": 500000,
    "log_interval": 5000,
    "win_reward": 198,  # CartPole-v0
    "win_break": True,
    "action_dim": None,
    "state_dim": None,
    "train": True,
    "test": True,
    "model_path": "\\",
}

env = gym.make(config.env)
config.action_dim = env.action_space.n
config.state_dim = env.observation_space.shape
agent = CNNDQNAgent(config)

# train
if config.train:
    trainer = Trainer(agent, env, config)
    trainer.train()

# test
elif config.test:
    tester = Tester(agent, env, config.model_path)
    tester.test()

# retrain
elif config.retrain:
    fr = agent.load_checkpoint(config.model_path + "checkpoint.pt")
    trainer = Trainer(agent, env, config)
    trainer.train(fr)
