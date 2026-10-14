import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

name = sys.argv[1]
model = PPO.load(name, device="cpu")
env = gym.make("LunarLander-v3")

max_x, max_tilt = [], []
for ep in range(20):
    obs, _ = env.reset(seed=ep)      # same starts for every model
    mx, mt = 0.0, 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        mx = max(mx, abs(float(obs[0])))
        mt = max(mt, abs(float(obs[4])))
        done = term or trunc
    max_x.append(mx)
    max_tilt.append(mt)

print(f"{name}: max sideways offset {np.mean(max_x):.2f}, "
      f"max tilt {np.degrees(np.mean(max_tilt)):.1f} deg (avg over 20 episodes)")