# MDP2 pick-and-place example (for review; removed before merging)

A Franka with an external and a wrist camera puts a cube onto a plate. It shows the MDP2 features that Isaac Lab
Arena needs, on the shape of Arena's openpi DROID evaluation:

- cameras rendered inside the step's CUDA graph, observed as `uint8` images in a group of separate terms;
- the DROID contract of openpi's adapter: `policy/joint_pos` (7), `policy/gripper_pos` (0 open, 1 closed),
  `camera_obs/external_camera_rgb`, `camera_obs/wrist_camera_rgb`, and 8 actions (7 absolute arm joint positions and
  a gripper command, closed above 0.5);
- a task config composed from a scene, an embodiment, and a task with `mdp2.compose`;
- success as a `sequence` of lifted, then placed; an episode budget with per-episode records;
- an object layout drawn from a pool that the term's `init` fills once.

Run its test from the repository root:

```bash
uv run --with pytest python -m pytest examples/mdp2_pick_place/test_pick_place.py
```

Or evaluate a policy:

```python
from isaaclab_experimental.envs import MDP2RLEnv
from pick_place_env_cfg import PickPlaceEnvCfg

cfg = PickPlaceEnvCfg()
cfg.episode_budget = 10
env = MDP2RLEnv(cfg)
obs, _ = env.reset()
while env.episodes_completed < cfg.episode_budget:
    obs, *_ = env.step(policy(obs))  # obs["policy"]["joint_pos"], obs["camera_obs"]["wrist_camera_rgb"], ...
print(env.episode_records()["success"].mean())
```
