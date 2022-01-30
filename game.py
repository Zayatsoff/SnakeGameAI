import pygame
import random
import config as c


class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.game_display = pygame.display.set_mode((c.WIDTH, c.HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.start_game()
        self.main_loop()

    def print_score(self, score):
        self.text = pygame.font.SysFont(c.SCORE_FONT[0], c.SCORE_FONT[1]).render(
            "Score: " + str(score), True, pygame.Color(c.SCORE_COL)
        )
        self.game_display.blit(self.text, [10, 10])

    # def debug(self):
    #     self.bug = pygame.font.SysFont(c.SCORE_FONT[0], c.SCORE_FONT[1]).render(
    #         "X: "
    #         + str(self.x)
    #         + "   TX: "
    #         + str(self.target_x)
    #         + "Y: "
    #         + str(self.y)
    #         + "   TY: "
    #         + str(self.target_y),
    #         True,
    #         pygame.Color(c.SCORE_COL),
    #     )
    #     self.game_display.blit(self.bug, [50, 50])

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

    def main_loop(self):
        # main game loop
        while not self.game_over:

            # restarts game
            while self.restart:
                Game()

            # main loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
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
            if self.x >= c.WIDTH or self.x < 0 or self.y >= c.HEIGHT or self.y < 0:
                self.restart = True

            self.x += self.x_speed
            self.y += self.y_speed

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

            # if snake crashes into itself
            for pixel in self.snake_pixels[:-1]:
                if pixel == [self.x, self.y]:
                    self.restart = True  # TODO: Restart

            self.draw_snake(c.SNAKE_SIZE, self.snake_pixels)
            self.print_score(self.snake_length - 1)

            # if snake eats food
            if self.x == self.target_x and self.y == self.target_y:
                self.spawn_food()
                self.snake_length += 1

            # self.debug()
            pygame.display.update()
            self.clock.tick(c.SNAKE_SPEED)

        pygame.quit()
        quit()


Game()
