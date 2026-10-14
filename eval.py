import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Model name: pass it in the terminal, or it defaults to ppo_baseline
name = sys.argv[1] if len(sys.argv) > 1 else "ppo_baseline"
model = PPO.load(name, device="cpu")

# Score on the PLAIN game (no custom reward), 20 attempts
plain = Monitor(gym.make("LunarLander-v3"))
mean, std = evaluate_policy(model, plain, n_eval_episodes=20)
print(f"{name}: {mean:.1f} +/- {std:.1f}")

# Record 3 attempts, each model gets its own folder
env = RecordVideo(
    gym.make("LunarLander-v3", render_mode="rgb_array"),
    f"videos/{name}",
    name_prefix=name,
    episode_trigger=lambda i: True,
)
for _ in range(3):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
env.close()
print("videos saved in", f"videos/{name}")