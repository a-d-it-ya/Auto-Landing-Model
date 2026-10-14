import gymnasium as gym

class ShapedReward(gym.Wrapper):
    """mode: 'fee' | 'hack' | 'fix'"""
    def __init__(self, env, mode="fee"):
        super().__init__(env)
        self.mode = mode
        self.prev_phi = 0.0

    @staticmethod
    def phi(o):
        # how good is this state: near the pad, slow, upright
        return -(abs(o[0]) + abs(o[1]) + abs(o[2]) + abs(o[3]) + abs(o[4]))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.prev_phi = self.phi(obs)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        r = float(r)

        if self.mode == "fee":      # small charge for every frame
            r -= 0.02

        elif self.mode == "hack":   # pay for sitting in a nice state
            if abs(obs[4]) < 0.1 and abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2:
                r += 1.0

        elif self.mode == "fix":    # pay for improvement + waiting costs
            new_phi = self.phi(obs)
            r += 5.0 * (new_phi - self.prev_phi) - 0.05
            self.prev_phi = new_phi

        return obs, r, term, trunc, info