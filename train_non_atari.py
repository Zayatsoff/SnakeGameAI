import torch
from collections import deque
import itertools
import numpy as np
import random
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from dqn import Network
from msgpack_numpy import patch as msgpack_numpy_patch
from torch.utils.tensorboard import SummaryWriter

from pytorch_wrappers import (
    BatchedPytorchFrameStack,
    PytorchLazyFrames,
    TransposeImageObs,
    make_deepmind,
)

from sneks.envs.snek import SingleSnek

config = {
    "GAMMA": 0.99,
    "BATCH_SIZE": 32,
    "BUFFER_SIZE": 1_000_000,
    "MIN_REPLAY_SIZE": 50_000,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.1,
    "EPSILON_DECAY": 1_000_000,
    "NUM_ENVS": 4,  # steps between gradient steps,
    "TARGET_UPDATE_FREQ": 10_000 / 4,
    "LR": 5e-5,
    "SAVE_PATH_EVERY": 10_000,
    "SAVE_PATH": "./saves/SingleSnekpack.pack",
    "LOG_DIR": "./logs/SingleSnek",
    "LOG_EVERY": 1_000,
    "LOAD": False,
    "DEBUGG": True,
}
msgpack_numpy_patch()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    make_env = lambda: Monitor(
        make_deepmind(SingleSnek(obs_type="rgb", n_food=1), scale_values=True),
        allow_early_resets=True,
    )

    # Dummy runs the env in sequence whilst Subproc runs the env in parallel. Both reset env at the end.
    vec_env = (
        DummyVecEnv([make_env for _ in range(config["NUM_ENVS"])])
        if config["DEBUGG"]
        else SubprocVecEnv([make_env for _ in range(config["NUM_ENVS"])])
    )
    env = BatchedPytorchFrameStack(vec_env, k=4)

    replay_buffer = deque(maxlen=config["BUFFER_SIZE"])
    epinfos_buffer = deque([], maxlen=100)

    episode_count = 0
    summary_writer = SummaryWriter(config["LOG_DIR"])

    online_net = Network(env, device=device, config=config)
    target_net = Network(env, device=device, config=config)

    online_net = online_net.to(device)
    target_net = target_net.to(device)
    if config["LOAD"]:
        online_net.load("./atari_model.pack")
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=config["LR"])

    # Initialize replay buffer
    obses = env.reset()
    for _ in range(config["MIN_REPLAY_SIZE"]):
        actions = [env.action_space.sample() for _ in range(config["NUM_ENVS"])]

        new_obses, rews, dones, infos = env.step(actions)
        for obs, action, rew, done, new_obs in zip(
            obses, actions, rews, dones, new_obses
        ):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

        obses = new_obses

    # Main Training Loop
    obs = env.reset()
    for step in itertools.count():
        epsilon = np.interp(
            step * config["NUM_ENVS"],
            [0, config["EPSILON_DECAY"]],
            [config["EPSILON_START"], config["EPSILON_END"]],
        )

        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)

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
        transitions = random.sample(replay_buffer, config["BATCH_SIZE"])
        loss = online_net.compute_loss(transitions, target_net)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Net
        if step % config["TARGET_UPDATE_FREQ"] == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % config["LOG_EVERY"] == 0:
            rew_mean = np.mean([e["r"] for e in epinfos_buffer]) or 0
            len_mean = np.mean([e["l"] for e in epinfos_buffer]) or 0
            print()
            print("Step:", step)
            print("Avg Rew:", rew_mean)
            print("Avg Ep Len:", len_mean)
            print("Episodes:", episode_count)

            summary_writer.add_scalar("Avg Reward", rew_mean, global_step=step)
            summary_writer.add_scalar("Avg Episode Len", len_mean, global_step=step)
            summary_writer.add_scalar("Episodes", episode_count, global_step=step)

        # Save
        if step % config["SAVE_PATH_EVERY"] == 0:
            print("Saving...")
            online_net.save(config["SAVE_PATH"])
