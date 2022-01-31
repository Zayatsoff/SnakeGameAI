import pygame
import random
import config as c
import numpy as np


class SnakeGame:
    def __init__(self, player="agent", action=None):
        self.clock = pygame.time.Clock()
        self.init_render = False
        self.start_game()
        self.main_loop(player, action)

    def print_score(self, score):
        self.text = pygame.font.SysFont(c.SCORE_FONT[0], c.SCORE_FONT[1]).render(
            "Score: " + str(score), True, pygame.Color(c.SCORE_COL)
        )
        self.game_display.blit(self.text, [10, 10])

    def draw_snake(self, snake_size, snake_pixels):
        for pixel in snake_pixels:
            pygame.draw.rect(
                self.game_display,
                pygame.Color(c.SNAKE_COL),
                (pixel[0], pixel[1], snake_size, snake_size),
            )

    def start_game(self):
        self.game_over = False
        self.restart = False

        self.x = c.WIDTH / 2
        self.y = c.HEIGHT / 2
        self.x_speed = 0
        self.y_speed = 0

        self.snake_pixels = []
        self.snake_length = 1

        self.spawn_food()
        pygame.init()

    def spawn_food(self):
        self.target_x = round(random.randrange(0, c.WIDTH - c.SNAKE_SIZE) / 10.0) * 10.0
        self.target_y = (
            round(random.randrange(0, c.HEIGHT - c.SNAKE_SIZE) / 10.0) * 10.0
        )

    def reset(self):
        self.init_render = False
        self.clock = pygame.time.Clock()
        self.game_display = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.start_game()

    def make_action(self, player, event, action):
        if player == "human":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.x_speed = -c.SNAKE_SIZE
                    self.y_speed = 0
                if event.key == pygame.K_d:
                    self.x_speed = c.SNAKE_SIZE
                    self.y_speed = 0
                if event.key == pygame.K_w:
                    self.x_speed = 0
                    self.y_speed = -c.SNAKE_SIZE
                if event.key == pygame.K_s:
                    self.x_speed = 0
                    self.y_speed = c.SNAKE_SIZE
        elif player == "agent":
            if action == 0:
                self.x_speed = -c.SNAKE_SIZE
                self.y_speed = 0
            if action == 1:
                self.x_speed = c.SNAKE_SIZE
                self.y_speed = 0
            if action == 2:
                self.x_speed = 0
                self.y_speed = -c.SNAKE_SIZE
            if action == 3:
                self.x_speed = 0
                self.y_speed = c.SNAKE_SIZE

    def check_food(self):
        if self.x == self.target_x and self.y == self.target_y:
            return True
        else:
            return False

    def get_state(self):

        obs = np.zeros((c.WIDTH, c.HEIGHT, 3), dtype="bool")

        obs[self.x, self.y, 0] = True

        for part in self.self.snake_pixels:
            obs[int(part[0]), int(part[1]), 1] = True

        obs[self.target_x, self.target_y, 2] = True

        return obs

    def check_death(self):
        for pixel in self.snake_pixels[:-1]:
            if pixel == [self.x, self.y]:
                return True
        if self.x >= c.WIDTH or self.x < 0 or self.y >= c.HEIGHT or self.y < 0:
            return True

        else:
            return False

    def render(self, mode="human"):
        if mode == "human":
            if not self.init_render:
                self.game_display = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
                pygame.display.set_caption("Snake Game")
                self.init_render = True

            # TODO: new score vs old score

            # bg colour
            self.game_display.fill(pygame.Color(c.BG_COL))

            # draw food
            pygame.draw.rect(
                self.game_display,
                pygame.Color(c.FOOD_COL),
                [self.target_x, self.target_y, c.SNAKE_SIZE, c.SNAKE_SIZE],
            )

            # let snake move
            self.snake_pixels.append([self.x, self.y])
            if len(self.snake_pixels) > self.snake_length:
                del self.snake_pixels[0]

            self.draw_snake(c.SNAKE_SIZE, self.snake_pixels)
            self.print_score(self.snake_length - 1)
            pygame.display.update()
            self.clock.tick(c.SNAKE_SPEED)

        elif mode == "print":
            if not self.init_render:
                print("Starting a new game!")
                self.init_render = True

            # if self.score > self.old_score:
            print("Found apple! Current score: " + str(self.score))

            # if self.score < self.old_score:
            #     print("Finished game with score: " + str(self.old_score))

            self.old_score = self.score

    def main_loop(self, player, action):
        # main game loop
        while not self.game_over:

            # restarts game
            while self.restart:
                self.reset()

            # main loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                self.make_action(player, event, action)

            self.x += self.x_speed
            self.y += self.y_speed

            # check death
            if self.check_death():
                self.restart = True

            # if snake eats food
            if self.check_food():
                self.spawn_food()
                self.snake_length += 1

            self.render()

        pygame.quit()
        quit()


# SnakeGame(player="human")
