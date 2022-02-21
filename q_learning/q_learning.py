import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 5000


EPSILON = 0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
EPSILON_DECAY_VALUES = EPSILON / (END_EPSILON_DECAY - START_EPSILON_DECAY)


DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_obs_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n])
)

ep_rewards = []
aggr_ep_reward = {"ep": [], "avg": [], "min": [], "max": []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print("Episode: ", episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:

        if np.random.random() > EPSILON:
            action = np.argmax(q_table[env.action_space.n])
        action = action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print(f"Objective reached on episode {episode}")

        discrete_state = new_discrete_state

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        EPSILON -= EPSILON_DECAY_VALUES

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_reward["ep"].append(episode)
        aggr_ep_reward["avg"].append(average_reward)
        aggr_ep_reward["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_reward["max"].append(max(ep_rewards[-SHOW_EVERY:]))

        print(
            f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}"
        )
env.close()

plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["avg"], label="avg")
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["min"], label="min")
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["max"], label="max")
plt.legend(loc=4)
plt.show()
