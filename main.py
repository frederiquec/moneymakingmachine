import pandas as pd
import gymnasium
import gym_trading_env

# df = pd.read_csv('EURUSD_M1.csv')
# print(df.head())

env = gymnasium.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(10):
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)

        if done or truncated:
            observation, info = env.reset()
env.close()

# with open('EURUSD_M1.csv') as f:
#     reader = list(csv.reader(f))
#     for row in reader[:10]:
#         row = list(map(float, row[1:]))
#         print(row)
