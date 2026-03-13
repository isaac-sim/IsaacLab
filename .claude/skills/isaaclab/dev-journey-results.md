# Dev Journey Results — 2026-02-25 (Run 2)

## Summary

Re-ran complete first-time user dev journey on GB10 aarch64 (Isaac Sim 6.0.0-rc.13,
Isaac Lab 4.0.0, develop branch). Tested ALL 4 RL frameworks end-to-end. Focused on
perceptive models / on-policy training path.

**Overall: All 4 frameworks train CartPole successfully. 1 known blocking bug (Direct
CartPole). 17 gaps identified (3 new, 14 confirmed from Run 1, 2 fixed since Run 1).**

## What Worked Well

1. **All 4 frameworks train CartPole** — RSL-RL (37s), SKRL (55s), RL-Games (61s), SB3 (103s)
2. **Checkpoints saved correctly** for all 4 frameworks
3. **TensorBoard event files** produced by all 4 frameworks, all readable
4. **Play script** loads trained policy and runs inference correctly
5. **Skill documents** accurately reflect codebase — verified against live code
6. **Agent configs** match documentation across all 4 frameworks
7. **Loss function locations** documented and verified via `inspect.getsourcefile()`
8. **Environment verification flow** works smoothly with `importlib.metadata`
9. **Fresh install guide** (setup/fresh-install.md) covers Kit vs source-build correctly

## Performance Benchmarks (GB10 aarch64, headless)

| Framework | Task | num_envs | Time | Final Reward | Steps/s |
|-----------|------|----------|------|-------------|---------|
| RSL-RL | Isaac-Cartpole-v0 | 4096 | 37s | 4.91 | ~265K |
| SKRL | Isaac-Cartpole-v0 | 4096 | 55s | ~4.9 | ~45 it/s |
| RL-Games | Isaac-Cartpole-v0 | 4096 | 61s | 4.89 | ~146K fps |
| SB3 | Isaac-Cartpole-v0 | 64 | 103s | 4.70 | ~5K |
| RSL-RL | Isaac-Cartpole-Direct-v0 | 4096 | CRASH | - | - |

## Blocking Issues

### BUG: Direct CartPole broken (develop branch) — STILL PRESENT
- **Symptom**: `RuntimeError: Cannot cast dtypes of unequal byte size`
- **Location**: `cartpole_env.py` → `write_root_pose_to_sim()` → warp `types.py`
- **Affects**: ALL frameworks on Isaac-Cartpole-Direct-v0
- **Status**: Same as Run 1, not fixed

## Gaps in Skill Documents

### New Gaps (found in Run 2)

1. **SKRL doesn't print reward metrics to stdout** (tutorials/cartpole.md, training/guide.md)
   - SKRL only shows a progress bar during training, no reward/loss info in terminal
   - First-time user sees `0%|...| 0/2400` → `100%|████| 2400/2400` with no convergence info
   - Must check TensorBoard to see if training converged
   - **Fix**: Add note: "SKRL shows only a progress bar. Use TensorBoard to monitor training."
   - **Status**: FIXING NOW

2. **SB3 warns about batch_size vs n_steps*n_envs mismatch** (training/sb3-reference.md)
   - CartPole config has `batch_size: 4096` but `n_steps=16 * n_envs=64 = 1024`
   - SB3 warns: "mini-batch size of 4096, but RolloutBuffer is of size 1024"
   - Safe to ignore but confusing for first-time users
   - **Fix**: Add to troubleshooting.md
   - **Status**: FIXING NOW

3. **`constant_fn()` deprecation warning from SB3** (troubleshooting.md)
   - "constant_fn() is deprecated, please use ConstantSchedule() instead"
   - Safe to ignore, comes from Isaac Lab's SB3 wrapper
   - **Fix**: Add to troubleshooting.md
   - **Status**: FIXING NOW

### Confirmed Gaps (from Run 1, still present)

4. **TensorBoard not on system PATH** — Documented in skills ✓ (fixed in Run 1)
5. **TensorBoard metric names differ** — Documented in skills ✓ (fixed in Run 1)
6. **Direct CartPole broken** — Documented in skills ✓ (warning added in Run 1)
7. **Loss function locations** — Documented in skills ✓ (added in Run 1)
8. **RSL-RL verification** — Now included in verification.md ✓ (fixed in Run 1)
9. **`rsl_rl.__version__` doesn't exist** — Documented in skills ✓ (fixed in Run 1)
10. **`pip list` returns nothing** — Documented in skills ✓ (fixed in Run 1)
11. **Deep imports require AppLauncher** — Documented in skills ✓ (fixed in Run 1)
12. **Play script runs infinitely** — Documented in skills ✓ (fixed in Run 1)
13. **Gym deprecation warning** — Documented in troubleshooting ✓ (fixed in Run 1)
14. **TensorFlow warning from TensorBoard** — Documented in troubleshooting ✓ (fixed in Run 1)
15. **PyTorch compute capability warning** — Documented in troubleshooting ✓ (fixed in Run 1)
16. **RayCaster variants undocumented** — Now documented in sensors-actuators.md ✓ (fixed in Run 1)
17. **Camera vs TiledCamera distinction** — Now documented in sensors-actuators.md ✓ (fixed in Run 1)

### Still Unfixed (low priority, carried forward)

- **`rlinf` framework lightly documented** — Only mentioned in training/guide.md
- **RL-Games config not as detailed as SB3/RSL-RL** — rl-games-reference.md exists but shorter

## Test Results Table

| Phase | Test | Result | Time |
|-------|------|--------|------|
| 1 | GPU/CUDA/Python detection | PASS | instant |
| 1 | python.sh wrapper | PASS | instant |
| 1 | PyTorch + CUDA import | PASS | 2s |
| 1 | All 4 RL framework imports | PASS | 3s |
| 1 | All 9 Isaac Lab packages installed | PASS | 2s |
| 2 | _isaac_sim symlink valid | PASS | instant |
| 2 | isaaclab.sh --help | PASS | instant |
| 2 | isaaclab.sh --new (template) | SKIP | Requires TTY |
| 2 | AppLauncher import | PASS | 2s |
| 2 | TensorBoard import | PASS | 1s |
| 3 | All 4 train.py --help | PASS | 4x2s |
| 3 | Agent configs readable | PASS | instant |
| 3 | Loss function locations verified | PASS | 3s |
| 4 | RSL-RL CartPole-v0 train | **PASS** | 37s |
| 4 | SB3 CartPole-v0 train | **PASS** | 103s |
| 4 | SKRL CartPole-v0 train | **PASS** | 55s |
| 4 | RL-Games CartPole-v0 train | **PASS** | 61s |
| 4 | RSL-RL play.py | PASS | runs OK |
| 4 | TensorBoard reads all 4 logs | PASS | 1s |
| 4 | Checkpoint files saved | PASS | instant |
| 4 | CartPole-Direct-v0 (RSL-RL) | **FAIL** | crash |

## Checkpoint Structure (all 4 frameworks)

### RSL-RL
```
logs/rsl_rl/cartpole/{timestamp}/
├── events.out.tfevents.* (TensorBoard)
├── model_0.pt, model_50.pt, model_100.pt, model_149.pt
├── params/env.yaml, agent.yaml
├── git/IsaacLab.diff
└── exported/ (ONNX export if requested)
```

### SB3
```
logs/sb3/Isaac-Cartpole-v0/{timestamp}/
├── PPO_1/events.out.tfevents.* (TensorBoard)
├── model.zip (final), model_*_steps.zip (periodic)
├── params/env.yaml, agent.yaml
└── command.txt (exact CLI used)
```

### SKRL
```
logs/skrl/cartpole/{timestamp}_ppo_torch/
├── events.out.tfevents.* (TensorBoard)
├── checkpoints/agent_*.pt, best_agent.pt
└── params/env.yaml, agent.yaml
```

### RL-Games
```
logs/rl_games/cartpole/{timestamp}/
├── summaries/events.out.tfevents.* (TensorBoard)
├── nn/cartpole.pth (best), last_cartpole_ep_*_rew_*.pth (periodic)
└── params/env.yaml, agent.yaml
```

## Key TensorBoard Metrics (verified with actual log data)

| Framework | Final Reward Tag | Final Value |
|-----------|-----------------|-------------|
| RSL-RL | `Train/mean_reward` | 4.91 |
| SB3 | `rollout/ep_rew_mean` | 4.70 |
| SKRL | `Reward / Total reward (mean)` | ~4.9 |
| RL-Games | `rewards/iter` | 4.89 |
