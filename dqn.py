from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
from baselines_wrappers.dummy_vec_env import DummyVecEnv
from baselines_wrappers.monitor import Monitor
from baselines_wrappers.subproc_vec_env import SubprocVecEnv

from pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames, make_atari_deepmind

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
NUM_ENVS = 4  # steps between gradient steps

class CNNDQN(nn.Module):
    def __init__(self, obs_space, depths=(32,64,64, final_layer=512)):
        super(CNNDQN, self).__init__()
        self.n_channels = obs.space.shape[0]
        cnndqn = nn.Sequential(
            nn.Conv2d(n_channels, depth[0], kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(n_channels, depth[1], kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(n_channels, depth[2], kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.num_actions = env.action_spaces.n
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(), nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obses_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample == epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions

    def compute_loss(self, transitions, target_net):
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss


make_env = lambda: Monitor(make_atari_deepmind("Breakout-v0"), allow_early_resets=True)
vec_env = DummyVecEnv(
    [make_env for _ in range(NUM_ENVS)]
)  # runs the env in sequence, debugging done here
# env = SubprocVecEnv([make_env for _ in NUM_ENVS])  # runs the env in parallel, training done here
# both reset env at the end

env = BatchedPytorchFrameStack(vec_env, k=4)

replay_buffer = deque(maxlen=BUFFER_SIZE)
epinfos_buffer = deque([0.0], maxlen=100)

episode_count = 0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize replay buffer
obses = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

    new_obses, rews, dones, infos = env.step(actions)
    for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

    obses = new_obses


# Main Training Loop
obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(
        step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]
    )

    if isinstance(obses[0], PytorchLazyFrames):
        act_obses = np.stack([o.get_frames() for o in obses])
        actions = online_net.act(obses, epsilon)

    else:
        actions = online_net.act(obses, epsilon)


    new_obses, rews, dones, infos = env.step(actions)
    for obs, action, rew, done, new_obs, info in zip(
        obses, actions, rews, dones, new_obses, infos
    ):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
        if done:
            epinfos_buffer.append(info["episode"])
            episode_count += 1

    obses = new_obses

    # start gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE)
    loss = online_net.comput_loss(transitions, target_net)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Net
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        rew_mean = np.mean([e["r"] for e in epinfos_buffer]) or 0
        len_mean = np.mean([e["l"] for e in epinfos_buffer]) or 0
        print()
        print("Step:", step)
        print("Avg Rew:", rew_mean)
        print("Avg Ep Len:", len_mean)
        print("Episodes:", episode_count)
