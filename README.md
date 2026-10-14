# Auto-Landing Model: Reward Shaping on Lunar Lander

I trained a PPO agent to land a rocket in Gymnasium's `LunarLander-v3`, then changed **only the reward** to show how reward design controls behavior. A badly designed reward made the agent hover forever instead of landing (reward hacking). A better-designed one fixed it.

Same rocket, same network, same learning algorithm. Different points, different behavior.

## Results

Mean score over 20 episodes, evaluated on the **unmodified** game (my reward wrapper is switched off at test time, so my own bonuses can't inflate the score). Each model trained for 1M steps. A score of 200+ counts as solved.

| Mode | Reward | Seed 0 | Seed 1 | Behavior |
|---|---|---|---|---|
| baseline | built-in reward | 261.1 ± 14.2 | 251.4 ± 47.8 | Lands |
| fee | built-in, minus 0.02 per frame | 276.3 ± 17.9 | not run | Lands |
| **hack** | built-in, plus 1 per frame while upright and slow | **-69.7 ± 20.0** | **-64.7 ± 25.3** | **Hovers, never lands** |
| fix | built-in, plus improvement bonus, minus 0.05 per frame | 268.0 ± 17.9 | 265.1 ± 26.3 | Lands |

The hack collapses from about 260 to about -65 on both seeds, and the fix recovers to about 265 on both seeds.

### Clips

| Model | Clips |
|---|---|
| Baseline (lands) | [`videos/ppo_baseline`](videos/ppo_baseline) |
| Hack (hovers) | [`videos/ppo_hack`](videos/ppo_hack), [`videos/ppo_hack_s1`](videos/ppo_hack_s1) |
| Fix (lands) | [`videos/ppo_fix`](videos/ppo_fix), [`videos/ppo_fix_s1`](videos/ppo_fix_s1) |

## How it works

**The environment.** Each frame the game gives the agent 8 numbers: horizontal position, height, horizontal speed, vertical speed, tilt, spin rate, and whether each leg is touching the ground. The agent picks one of 4 actions per frame: do nothing, fire the left engine, fire the main engine, or fire the right engine.

**The agent.** PPO (Proximal Policy Optimization) from Stable-Baselines3 with a small MLP (two layers of 64 units). It plays thousands of attempts, sees which choices led to higher total reward, and nudges its weights toward them in small, stable steps.

**The reward.** The game's built-in reward pays for getting closer, slower and more upright, charges a little for fuel, and gives +100 for a safe landing or -100 for a crash. My wrapper (`shaped.py`) sits between the game and the agent, takes that reward, and adds or subtracts its own terms.

## The experiment

### 1. Hack: pay for sitting in a nice state

```python
if abs(obs[4]) < 0.1 and abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2:
    r += 1.0   # upright and slow, +1 every frame
```

An attempt is cut off at 1,000 frames, so a rocket that hovers upright and slow can collect hundreds of points. Landing ends the attempt and stops the payments, so hovering beats landing. The agent learned exactly that. On the plain game it scores about -65 to -70, because it never earns the +100 landing bonus and keeps burning fuel.

The agent doesn't learn what you meant. It learns what you scored.

### 2. Fix: pay for improvement, and make waiting cost something

```python
phi = -(abs(x) + abs(y) + abs(vx) + abs(vy) + abs(angle))   # how good is this state
r += 5.0 * (phi_new - phi_old) - 0.05
```

The bonus is the **change** in how good the state is between frames, not the state itself. Hovering produces no change, so it earns nothing, and the per-frame fee makes every extra frame cost points. The agent went back to landing on both seeds.

### 3. Does the fix make landings sloppier?

I noticed some fix-run clips swayed more before landing, so I measured it instead of trusting my eyes. Max tilt and max sideways offset over 20 episodes with identical starting conditions:

| Model | Max sideways offset | Max tilt |
|---|---|---|
| baseline, seed 0 | 0.19 | 11.9° |
| baseline, seed 1 | 0.17 | 15.0° |
| fix, seed 0 | 0.18 | 11.7° |
| fix, seed 1 | 0.19 | 11.6° |

No measurable difference between the plain reward and the fix. The swaying I saw in a few clips was down to the random starting conditions of the 3 recorded episodes, not the reward.

## Repo structure

| File | What it does |
|---|---|
| `random_agent.py` | Watch an untrained agent crash, and print the 8 observation numbers |
| `train.py` | Train the baseline PPO agent with the built-in reward |
| `shaped.py` | `ShapedReward` wrapper with the `fee`, `hack` and `fix` modes |
| `train_shaped.py` | Train with a chosen mode and seed |
| `eval.py` | Score a model over 20 episodes on the plain game and record 3 clips |
| `eval_style.py` | Measure max tilt and sideways offset over 20 seeded episodes |

## Run it yourself

Requires Python 3.10 or newer. A GPU is not needed, and the CPU is actually faster for a network this small. Each training run took about 15-20 minutes on a laptop.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install swig
pip install "gymnasium[box2d]" stable-baselines3 moviepy
```

If the Box2D install fails with an error mentioning `swig`, install `swig` first and then the `box2d` extra again.

```bash
python random_agent.py             # watch a random agent crash

python train.py                    # baseline, saves ppo_baseline.zip
python train_shaped.py hack 1      # modes: base | fee | hack | fix, then a seed
python train_shaped.py fix 1

python eval.py ppo_hack_s1         # score and record clips to videos/ppo_hack_s1
python eval_style.py ppo_fix_s1    # tilt and sideways offset
```

`train_shaped.py` saves models as `ppo_<mode>_s<seed>.zip`. The mode `base` applies no shaping, so it gives a plain-reward run on a chosen seed.

## Limitations

- **Two seeds per mode.** That's enough to show the hack and the fix reproduce, but not enough to rank small differences. For example, fee (276) versus baseline (261) is within noise, so I don't claim the fee helped.
- **The seed-0 runs were unseeded.** They were trained with an earlier version of `train_shaped.py` that didn't set a seed and saved models as `ppo_<mode>.zip`, so they can't be reproduced exactly. The seed-1 runs use the current script and are reproducible.
- **The fee mode was only run once.** It was a sanity check that the wrapper plugs in correctly, not part of the main comparison.
- **The hack and fix terms are hand-tuned.** I didn't sweep the bonus size or the fee.

## Acknowledgements

The project structure follows Hari's ([@harixganesh](https://x.com/harixganesh)) guide to reward shaping on Lunar Lander. Built with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
