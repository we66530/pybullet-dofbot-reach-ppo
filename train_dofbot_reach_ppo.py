import os
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from dofbot_reach_env import DofbotReachEnv
from csv_logging_callback import CSVLoggerCallback


TOTAL_TIMESTEPS = 500_000
LOG_DIR = "./logs/dofbot_reach_ppo"
MODEL_DIR = "./models"
SEED = 42
N_ENVS = 8

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(rank, seed=0):
    def _init():
        env = DofbotReachEnv(
            render_mode=None,
            max_episode_steps=300,
            distance_threshold=0.03,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    env = SubprocVecEnv(
        [make_env(i, SEED) for i in range(N_ENVS)]
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    callback = CallbackList([
        CheckpointCallback(
            save_freq=50_000,
            save_path=MODEL_DIR,
            name_prefix="ppo_dofbot_reach",
        ),
        CSVLoggerCallback(
            log_dir=LOG_DIR,
            filename="training_log.csv",
        ),
    ])

    print("🚀 Training PPO (Dofbot Reach)")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_dofbot_reach_final"))
    env.close()
