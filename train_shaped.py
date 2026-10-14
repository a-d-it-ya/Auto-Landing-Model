import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from shaped import ShapedReward

MODE = sys.argv[1] if len(sys.argv) > 1 else "fee"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

env = make_vec_env("LunarLander-v3", n_envs=8, seed=SEED,
                   wrapper_class=ShapedReward, wrapper_kwargs={"mode": MODE})
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4,
            gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
            device="cpu", seed=SEED, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save(f"ppo_{MODE}_s{SEED}")
print("saved", MODE, SEED)