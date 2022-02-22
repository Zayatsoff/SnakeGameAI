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
from dqn import DQNAgent

"""Loss: mse, optim = Ama, lr=0.001, metric = accuracy"""
# LEARNING_RATE = 0.001
# REPLAY_MEMORY_SIZE = 50_000
# MIN_REPLAY_SIZE = 1_000
# MINIBTACH_SIZE = 64
# DISCOUNT = 0.99
# UPDATE_TARGET_EVERY = 5
EPISODES = 20_000
# EPSILON = 1.0
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -200
SHOW_PREVIEW = False

env = BlobEnv()

ep_rewards = [-200]

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if not os.path.isdir("models"):
    os.makedirs("models")

agent = DQNAgent()
for episode in tqdm(range(1, EPISODES), ascii=True, unit="episode"):
    # agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False

    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        new_state, reward, done = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
                ep_rewards[-AGGREGATE_STATS_EVERY:]
            )
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward,
                reward_min=min_reward,
                reward_max=max_reward,
                epsilon=EPSILON,
            )

            # Save model, but only when min reward is greater or equal a set value
            # if min_reward >= MIN_REWARD:
            #     agent.model.save(
            #         f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
            #     )

        # Decay epsilon
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)
