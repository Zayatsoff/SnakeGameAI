from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_INTERVAL = 1000


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tahn(), nn.Lnear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        axtion = max_q_index.detach().item
        return action


env = gym.make("CartPole-v0")

replay_buffer = deque(maxlen=BUFFER_SIZE)

reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

# init replay buffer
obs = env.restet()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, reward, done, _ = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

# main training loop
obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, reward, done, _ = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += reward
    if done:
        obs = env.reset()

        reward_buffer.append(episode_reward)
        episode_reward = 0.0

# start gradient step

transitions = random.sample(replay_buffer, BATCH_SIZE)
obses = np.asarray(t[0] for t in transitions)
actions = np.asarray(t[1] for t in transitions)
rewards = np.asarray(t[2] for t in transitions)
dones = np.asarray(t[3] for t in transitions)
new_obses = np.asarray(t[4] for t in transitions)

transitions_t = torch.as_tensor(obses, dtype=torch.float32)
obses_t = torch.as_tensor(obses, dtype=torch.float32)
actions_t = torch.as_tensor(obses, dtype=torch.int64)
rewards_t = torch.as_tensor(obses, dtype=torch.float32)
dones_t = torch.as_tensor(obses, dtype=torch.float32)
new_obses_t = torch.as_tensor(obses, dtype=torch.float32)

# compute targets
