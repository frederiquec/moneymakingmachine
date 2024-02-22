import pandas as pd
import gymnasium
import numpy as np
import gym_trading_env

# Load CSV into DataFrame
df = pd.read_csv('EURUSD_M1.csv', header=None, names=["timestamp", "open", "high", "low", "close", "volume"])

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
env = gymnasium.make("TradingEnv", df = df, positions = [-0.5, 0, 0.5], initial_position= 0, max_episode_duration = 'max', portfolio_initial_value = 1000000, verbose = 2)
env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )

for _ in range(1):
    done = False
    truncated = False
    observation, info = env.reset()
    while not done and not truncated:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
    if done:
        print("ENDED CUS DONEZO")
    if truncated: 
        print("ENDED CUS NO MORE DAATAA")
    env.unwrapped.save_for_render(dir="render_logs")
env.close()
