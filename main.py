import pandas as pd
import gymnasium
import gym_trading_env

# Load CSV into DataFrame
df = pd.read_csv('moneymakingmachine\EURUSD_M1.csv', header=None, names=["timestamp", "open", "high", "low", "close", "volume"])

# Print the columns to check if they match your expectations
print(df.columns)

# Calculate Required Features
df["feature_pct_change"] = df["close"].pct_change() 
df["feature_high"] = df["high"] / df["close"] - 1     
df["feature_low"] = df["low"] / df["close"] - 1       

# Preprocess DataFrame
df.dropna(inplace=True)

# Set up Gym Environment
env = gymnasium.make("TradingEnv", df=df, positions=[-1, 0, 1], initial_position=1)
observation, info = env.reset(seed=42)

for _ in range(10):
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
    
        if done or truncated:
            observation, info = env.reset()
    env.save_for_render(dir = "render_logs")
env.close()
