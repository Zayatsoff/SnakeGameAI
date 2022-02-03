from gym.envs.registration import register

register(
    id="Snake-v0",
    entry_point="gym_game.envs:SnakeEnv",
    max_episode_steps=2000,
)
