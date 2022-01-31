import time
import gym
from gym import spaces
import numpy as np
from snake_game import SnakeGame
import config as c


class SnakeEnv(gym.Env):
    """Open AI Snake Environment"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, max_without_eating=300):
        super(SnakeEnv, self).__init__()
        self.max_without_eating = max_without_eating
        self.steps_without_food = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(c.HEIGHT, c.WIDTH, 3), dtype=np.uint8
        )
        self.game = SnakeGame(player="agent")

    def step(self, action):
        self.game.main_loop(action)

        if self.game.check_food():
            reward = 1
            self.steps_without_food = 0
        elif self.game.check_death():
            reward = -1
        else:
            reward = 0
            self.steps_without_food += 1

        done = (
            self.steps_without_food > self.max_without_eating or self.game.check_death()
        )

        return self.game.get_state(), reward, done, {}

    def reset(self):
        self.game.reset()
        self.explored = np.zeros_like(self.explored)
        self.steps_without_food = 0
        return self.game.get_state()

    def render(self, mode="human", close=False):
        if mode == "human":
            time.sleep(0.1)
            self.game.render(mode=mode)

    def seed(self, seed=None):
        np.random.seed(seed)
