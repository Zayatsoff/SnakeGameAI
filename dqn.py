import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import tqdm
from utils import ReplayBuffer, save_checkpoint, load_checkpoint


class CNNDQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(CNNDQN, self).__init__()
        self.dqn = nn.Sequential(
            nn.Conv2d(num_inputs[0], 256, kernel_size=3),
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
        self.buffer = ReplayBuffer(self.config["max_buff"])

        self.model = CNNDQN(self.config["state_dim"], self.config["action_dim"])
        self.target_model = CNNDQN(self.config["state_dim"], self.config["action_dim"])
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        if self.config["use_cuda"]:
            self.cuda()

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def act(self, state, epsilon):
        if epsilon is None:
            epsilon = self.config["epsilon_min"]

        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config["use_cuda"]:
                state = state.cuda()

            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config["action_dim"])
        return action

    def learning(self, episode):
        state, action, reward, next_state, done = self.buffer.sample(
            self.config["batch_size"]
        )

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config["use_cuda"]:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()

        q_values = self.model(state).cuda()
        next_q_values = self.model(next_state).cuda()
        next_q_state_values = self.target_model(next_state).cuda()

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(
            1, next_q_values.max(1)[1].unsqueeze(1)
        ).squeeze(1)
        expected_q_value = reward + self.config["gamma"] * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if episode % self.config["update_tar_interval"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save_model(self, output, name=""):
        save_checkpoint(output, self.model_optim, filename="checkpoint_" + name + ".pt")

    def save_checkpoint(
        self,
        episode,
        output,
    ):
        save_checkpoint(
            output, self.model_optim, filename="checkpoint_" + episode + ".pt"
        )

    def load_weights(self, filename, model):
        load_checkpoint(filename, model, self.model_optim, self.config["learning_rate"])
