import pandas as pd
import gymnasium
import gym_trading_env

# Load CSV into DataFrame
df = pd.read_csv('moneymakingmachine\EURUSD_M1.csv', header=None, names=["timestamp", "open", "high", "low", "close", "volume"])

# Convert timestamp column to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate additional features
df["feature_pct_change"] = df["close"].pct_change() 
df["feature_high"] = df["high"] / df["close"] - 1     
df["feature_low"] = df["low"] / df["close"] - 1       

# Remove rows with NaN values
df.dropna(inplace=True)

# Create trading environment
env = gymnasium.make("TradingEnv", df=df, positions=[-1, 0, 1], initial_position='random', portfolio_initial_value=5, max_episode_duration=10000, verbose=1)

# Reset environment and get initial observation
observation, info = env.reset()

# Main loop
for _ in range(1):
    done = False
    truncated = False
    while not done and not truncated:
        action = -1  # Example action (you may want to change this)
        observation, reward, done, truncated, info = env.step(action)
    
        # Reset environment if episode is done or truncated
        if done or truncated:
            observation, info = env.reset()
    
    # Save rendering data with formatted datetime
    env.unwrapped.save_for_render(dir="render_logs")

# Close environment
env.close()
