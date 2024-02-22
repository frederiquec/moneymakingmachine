import pandas as pd
import gymnasium
import numpy as np
import gym_trading_env

# Load CSV into DataFrame
df = pd.read_csv('EURUSD_M1.csv', header=None, names=["timestamp", "open", "high", "low", "close", "volume"])

# Print the columns to check if they match your expectations
print(df.columns)

# Calculate Required Features
df["feature_pct_change"] = df["close"].pct_change() 
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"] / df["close"] - 1     
df["feature_low"] = df["low"] / df["close"] - 1   
df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()    

# Preprocess DataFrame
df.dropna(inplace=True)
df.reset_index(inplace=True)  # Reset index
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime

# Set up Gym Environment
env = gymnasium.make("TradingEnv", df=df, positions=[-1, 0, 1], initial_position='random', portfolio_initial_value=5, max_episode_duration=10000)


for _ in range(1):
    done = False
    truncated = False
    observation, info = env.reset()
    while not done and not truncated:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
    env.unwrapped.save_for_render(dir="render_logs")
env.close()
