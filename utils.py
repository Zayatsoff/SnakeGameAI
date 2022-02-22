import random
import numpy as np
import torch
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter

# General
def save_checkpoint(model, optimizer, filename="checkpoint.pt"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # set old lr to new lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_epoch_loss(model, data_loader, device):
    """
    Code taken from https://github.com/rasbt/deeplearning-models
    """
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def compute_accuracy(model, data_loader, device):
    """
    Code taken from https://github.com/rasbt/deeplearning-models
    """
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


class TensorBoardLogger(object):
    """
    Originally from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514, rewritten to work with torch tensorboard
    """

    def __init__(self, log_dir):
        """Create action summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, step, value):
        """Log action scalar variable."""
        self.writer.add_scalar(tag, value, step)


# DQN
class ReplayBuffer(object):
    """
    Code taken from  https://github.com/blackredscarf
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state[None, :], action, reward, next_state[None, :], done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def size(self):
        return len(self.buffer)


class train_dqn:
    """
    Adapted from  https://github.com/blackredscarf
    """

    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

        self.epsilon_final = self.config["epsilon_min"]
        self.epsilon_start = self.config["epsilon"]
        self.epsilon_decay = self.config["eps_decay"]
        self.epsilon = self.epsilon_start

        self.outdir = self.config["model_path"]
        self.logger = TensorBoardLogger(self.outdir)

    def train(self):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = True

        state = self.env.reset()
        for episode in range(1, self.config["episodes"]):
            action = self.agent.act(state, self.epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            loss = 0

            if self.agent.buffer.size() > self.config["batch_size"]:
                loss = self.agent.learning(episode)
                losses.append(loss)
                self.logger.scalar_summary("Loss per episode", episode, loss)

            if episode % self.config["print_interval"] == 0:
                print(
                    "episode: %5d, reward: %5f, loss: %4f episode: %4d"
                    % (episode, np.mean(all_rewards[-10:]), loss, episode)
                )

            if episode % self.config["log_interval"] == 0:
                self.logger.scalar_summary(
                    "Reward per episode", episode, all_rewards[-1]
                )  # TODO

            if (
                self.config["checkpoint"]
                and episode % self.config["checkpoint_interval"] == 0
            ):
                self.agent.save_checkpoint(self.agent, episode)

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.logger.scalar_summary(
                    "Best 100-episodes average reward", episode, avg_reward
                )

                if (
                    len(all_rewards) >= 100
                    and avg_reward >= self.config["win_reward"]
                    and all_rewards[-1] > self.config["win_reward"]
                ):
                    is_win = True
                    self.agent.save_model(self.outdir, "best")
                    print(
                        "Episodes in total: %d  \n100-episodes average reward: %3f \n%d trials"
                        % (episode, avg_reward, episode - 100)
                    )
                    if self.config["win_break"]:
                        break

        if not is_win:
            print("Did not solve after %d episodes" % episode)
            self.agent.save_model(self.outdir, "last")


class test_dqn:
    def __init__(
        self,
        agent,
        env,
        filename,
        config,
        num_episodes=50,
        max_ep_steps=400,
        test_ep_steps=100,
    ):
        self.num_episodes = num_episodes
        self.max_ep_steps = max_ep_steps
        self.test_ep_steps = test_ep_steps
        self.agent = agent
        self.env = env
        self.agent.is_training = False
        load_checkpoint(
            filename, self.agent, agent.model_optim, config["learning_rate"]
        )
        self.policy = lambda x: agent.act(x)

    def test(self, debug=False, render=True):
        avg_reward = 0
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_steps = 0
            episode_reward = 0.0

            done = False
            while not done:
                if render:
                    self.env.render()

                action = self.policy(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                if episode_steps + 1 > self.test_ep_steps:
                    done = True

            if debug:
                print(
                    "[Test] episode: %3d, episode_reward: %5f"
                    % (episode, episode_reward)
                )

            avg_reward += episode_reward
        avg_reward /= self.num_episodes
        print("avg reward: %5f" % (avg_reward))


# Classification
def train_classifier(
    num_epochs,
    model,
    optimizer,
    device,
    train_loader,
    valid_loader=None,
    loss_fn=None,
    logging_interval=100,
    skip_epoch_stats=False,
):

    # Modified from https://github.com/rasbt/deeplearning-models
    log_dict = {
        "train_loss_per_batch": [],
        "train_acc_per_epoch": [],
        "train_loss_per_epoch": [],
        "valid_acc_per_epoch": [],
        "valid_loss_per_epoch": [],
    }

    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict["train_loss_per_batch"].append(loss.item())

            if not batch_idx % logging_interval:
                print(
                    "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                    % (epoch + 1, num_epochs, batch_idx, len(train_loader), loss)
                )

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_acc = compute_accuracy(model, train_loader, device)
                train_loss = compute_epoch_loss(model, train_loader, device)
                print(
                    "***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f"
                    % (epoch + 1, num_epochs, train_acc, train_loss)
                )
                log_dict["train_loss_per_epoch"].append(train_loss.item())
                log_dict["train_acc_per_epoch"].append(train_acc.item())

                if valid_loader is not None:
                    valid_acc = compute_accuracy(model, valid_loader, device)
                    valid_loss = compute_epoch_loss(model, valid_loader, device)
                    print(
                        "***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f"
                        % (epoch + 1, num_epochs, valid_acc, valid_loss)
                    )
                    log_dict["valid_loss_per_epoch"].append(valid_loss.item())
                    log_dict["valid_acc_per_epoch"].append(valid_acc.item())

        print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))
        save_checkpoint(model, optimizer)
        print("Saved!")

    print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))

    return


def validate_classifier(
    model,
    device,
    valid_loader=None,
    loss_fn=None,
):

    log_dict = {
        "train_loss_per_batch": [],
        "train_acc_per_epoch": [],
        "train_loss_per_epoch": [],
        "valid_acc_per_epoch": [],
        "valid_loss_per_epoch": [],
    }

    if loss_fn is None:
        loss_fn = F.cross_entropy

    model.eval()

    with torch.set_grad_enabled(False):  # save memory during inference

        valid_acc = compute_accuracy(model, valid_loader, device)
        valid_loss = compute_epoch_loss(model, valid_loader, device)
        print("*** Valid. Acc.: %.3f%% | Loss: %.3f" % (valid_acc, valid_loss))
        log_dict["valid_loss_per_epoch"].append(valid_loss.item())
        log_dict["valid_acc_per_epoch"].append(valid_acc.item())

    return
