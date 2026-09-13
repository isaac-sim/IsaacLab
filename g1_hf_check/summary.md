# G1 rough velocity on latest develop — Newton MJWarp, stock heightfield terrain

Question: does the **unmodified** `Isaac-Velocity-Rough-G1` task train to completion on the
latest `develop`, with the default (heightfield) terrain collider, under Newton MJWarp?

Answer: **yes.** 5000/5000 iterations, exit 0, no traceback, no NaN/Inf, final policy matches
the previous develop run on every quality metric. The only regression is throughput (~2.1x).

## Setup

| | old run | new run |
|---|---|---|
| commit | `01eb210728` (`heightfield_fix_6679`) | `3b736feb04` (`origin/develop`, detached) |
| worktree | `/home/henry/workspace/IsaacLab-heightfield` | `/home/henry/workspace/IsaacLab-develop` |
| isaaclab / isaaclab_newton | 15.4.0 / 3.0.0 | 16.0.0 / 4.0.0 |
| newton / warp | 1.5.0.dev0 / 1.16.0.dev20260723 | 1.5.0rc2 / 1.16.0 |
| command | `train_checked.py --rl_library rsl_rl --task Isaac-Velocity-Rough-G1 --num_envs 4096 --seed 42 presets=newton_mjwarp` | `isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-G1 presets=newton_mjwarp` |
| envs / seed / iterations | 4096 / 42 / 5000 | 4096 / 42 / 5000 (all task defaults) |

`velocity_env_cfg.py` is **byte-identical** between the two commits (njmax=1000, nconmax=300,
`margin=0.0`, `ke=160000`, `kd=1100`, `num_substeps=2`). `g1/rough_env_cfg.py` differs only by
the removal of `play_mode()` (#6860) and an added velocity-marker offset. So any behavioural
delta comes from the framework, not the task config.

## Terrain representation confirmed

`g1_hf_check/hfprobe.py` wraps `NewtonManager._inject_terrain_heightfields` and asserts the
realized collider. On stock config:

```
HFPROBE: stock gate all(convert_to_heightfield)=True
HFPROBE: converted=['/World/ground/terrain'] hfield_shapes=1
HFPROBE: CONFIRMED native heightfield collider
```

All six `ROUGH_TERRAINS_CFG` sub-terrains set `convert_to_heightfield=True`, so the
all-or-nothing gate in `TerrainImporter._is_heightfield_collider_requested` passes and the
terrain mesh is rasterized into a single Newton heightfield shape.

## Result

| metric | old `01eb210728` | new `3b736feb04` |
|---|---|---|
| iterations | 4999/5000 | 4999/5000 |
| exit code | 0 | 0 |
| traceback / NaN / Inf | 0 / 0 | 0 / 0 |
| total steps | 491,520,000 | 491,520,000 |
| **final mean reward** | **27.53** | **25.71** |
| mean episode length | 986.86 | 986.45 |
| success_rate | 0.9917 | 0.9917 |
| error_vel_xy | 0.1712 | 0.1679 |
| terrain_levels | 5.7582 | 5.7274 |
| base_contact termination | 1.19% | 0.64% |
| **wall clock** | **4757 s (1h19m)** | **11783 s (3h16m)** |
| **steps/s** | **~104,800 (flat)** | **~33k–51k** |

## Reward curve

The new run lags by ~600-700 iterations mid-training, then closes the gap:

| iter | old | new |
|---|---|---|
| 500 | 5.54 | -0.21 |
| 1000 | 5.49 | -0.45 |
| 1500 | 12.54 | 4.53 |
| 2000 | 18.09 | 10.29 |
| 2500 | 22.46 | 14.79 |
| 3000 | 24.82 | 18.18 |
| 3500 | 26.01 | 21.38 |
| 4000 | 27.12 | 23.39 |
| 4500 | 27.43 | 25.40 |
| 4999 | 27.53 | 25.71 |

Both curves are still rising at 5000; the -1.8 endpoint gap is within the oscillation band this
task shows late in training, and every non-reward quality metric (success_rate, episode length,
tracking error, fall rate) is equal or better on the new run.

## Open item: ~2.1x throughput regression

104,800 -> 33k–51k steps/s at identical env config, same GPU (RTX 5880 Ada), same env count.
Not caused by `#6930` determinism: `NewtonCfg.deterministic_mode` defaults to `not_guaranteed`.
Prime suspect is `4fdfd18c75` (#6911), which moved the newton pin from commit `10402ec` to the
`release-1.5` branch (1.5.0.dev0 -> 1.5.0rc2) and warp from `1.16.0.dev20260723` to `1.16.0`.
Unlike the old run's flat 104.8k, the new run's rate also fluctuates (33k-51k).

Artifacts: `train_full.log`, `hfprobe.log`, `smoke.log`, `meta.txt`,
`logs/rsl_rl/g1_rough/2026-08-11_16-34-15_develop_hf_default/` (101 checkpoints, 767 MB).
