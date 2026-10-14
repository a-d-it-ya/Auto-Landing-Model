import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
for _ in range(500):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    print(obs.round(2), round(float(r), 2))
    if term or trunc:
        obs, _ = env.reset()
env.close()