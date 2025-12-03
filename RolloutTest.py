import time
import numpy as np
from dofbot_reach_env import DofbotReachEnv

env = DofbotReachEnv(render_mode="human")

obs, info = env.reset()

try:
    while True:
        # random action (just for visual check)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"dist={info['distance']:.3f}, reward={reward:.3f}",
            end="\r"
        )

        # ✅ 關鍵：讓 GUI 跟得上人類時間
        time.sleep(1.0 / 60.0)

        if terminated or truncated:
            print("\nEpisode finished, resetting...")
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nExit rollout test")
    env.close()
