import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""Loss: mse, optim = Ama, lr=0.001, metric = accuracy"""
LEARNING_RATE = 0.001


class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()
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


class DQNAgent:
    def __init__(self):
        # main model
        self.model = DQN()
        self.optim = optim.Adam(self.model.parameters, lr=LEARNING_RATE)
        self.loss_function = nn.MSELoss()
        # target model
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
