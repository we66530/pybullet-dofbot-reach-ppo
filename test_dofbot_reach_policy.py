import time
from stable_baselines3 import PPO
from dofbot_reach_env import DofbotReachEnv

model = PPO.load("./models/ppo_dofbot_reach_final")

env = DofbotReachEnv(render_mode="human")

obs, info = env.reset()

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(1.0 / 60.0)

        if terminated or truncated:
            obs, info = env.reset()

except KeyboardInterrupt:
    env.close()
