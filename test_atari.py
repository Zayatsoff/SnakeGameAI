import numpy as np
import torch
import itertools
from stable_baselines3.common.vec_env import DummyVecEnv
from dqn import Network
from pytorch_wrappers import (
    make_atari_deepmind,
    BatchedPytorchFrameStack,
    PytorchLazyFrames,
)
import time
from msgpack_numpy import patch as msgpack_numpy_patch
from train_atari import config

msgpack_numpy_patch()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

make_env = lambda: make_atari_deepmind("Breakout-v0")
vec_env = DummyVecEnv([make_env for _ in range(1)])
env = BatchedPytorchFrameStack(vec_env, k=4)

net = Network(env, device, config)
net = net.to(device)

net.load("./saves/atari_model.pack")

obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    env.render()
    time.sleep(0.02)

    if done[0]:
        obs = env.reset()
        beginning_episode = True
