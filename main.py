import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import tqdm
from dqn import CNNDQNAgent
from utils import train_dqn, test_dqn
import gym


# dictionairy
config = {
    "env": "CartPole-v0",
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "eps_decay": 375,
    "episodes": 25000,
    "use_cuda": True,
    "learning_rate": 1e-4,
    "max_buff": 100000,
    "update_tar_interval": 12,
    "batch_size": 32,
    "print_interval": 64,
    "checkpoint": True,
    "checkpoint_interval": 6400,
    "log_interval": 64,
    "win_reward": 198,  # CartPole-v0
    "win_break": True,
    "action_dim": None,
    "state_dim": None,
    "train": True,
    "test": True,
    "model_path": "\\",
}

env = gym.make(config["env"])
config["action_dim"] = env.action_space.n
config["state_dim"] = env.observation_space.shape
agent = CNNDQNAgent(config)

# train
if config["train"]:
    trainer = train_dqn(agent, env, config)
    trainer.train()

# test
elif config["test"]:
    tester = test_dqn(agent, env, "checkpoint_best.pt", config)
    tester.test()

# retrain
elif config["retrain"]:
    fr = agent.load_checkpoint("checkpoint.pt", agent)
    trainer = train_dqn(agent, env, config)
    trainer.train(fr)
