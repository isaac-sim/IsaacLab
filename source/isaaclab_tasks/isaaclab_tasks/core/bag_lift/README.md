# Isaac-BagLift-Direct

An SO-101 arm grasps a bag by its handle, lifts it and carries it home. Isaac Lab Direct workflow, rl_games PPO,
state observations (47 numbers), 30 Hz policy. One task, one reward file. Training is residual around the
`grasp_poses.json` sequence: a zero action replays that open-loop demo.

Policies trained before the v2 changes (39 observations, 60 Hz, `logs/rl_games/bag_lift_direct/`) do not load into
this env. The code they were trained with is saved next to the last of those runs, in
`logs/rl_games/bag_lift_direct/2026-10-31_12-01-22/code_snapshot/`.

The camera task, per-env lighting and the template-leftover agent configs were removed to keep this small.
A full copy of the previous version (cameras, lights, D415 utils) is in the Cursor agent store at
`bag_lift_backup_2026-10-29/` for when cameras come back.


Note to self:

Visualize
 uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit
 Edit: json files (grasp poses).




## Task definition

| Term | Definition (`BagLiftEnv._update_task_state`) |
|---|---|
| **grasp** | gripper commanded closed (`revolute_7` target within `grip_closed_margin` of its closed limit) AND the strap held geometrically. No force threshold. Either: **resting on the rigid jaw** (preferred: at least `min_strap_nodes` strap nodes within `pinch_dist` = 8 mm of the rigid jaw, on its mouth side, at depth below `mouth_mid_frac + mouth_tol` = 0.8), or **pinched between the jaws** (at least `min_strap_nodes` nodes within `pinch_dist` of both jaws, on opposite sides, at depth `mouth_mid_frac` 0.5 ± `mouth_tol` 0.3). Depth 0 = housing end of the rigid jaw (`mouth_base_y`), 1 = jaw tip (`mouth_tip_y`). Proxy points come from `assets/jaw_points.json`. **Debounced**: `grasped` turns on after `grasp_on_steps` (3) raw hits in a row and off after `grasp_off_steps` (15, 0.5 s) misses in a row |
| **lift** | grasped, bag at least `lift_height` (5 cm) above the height it settled at. A progress stage (reward + `lift rate`) |
| **final pose / success** | gripper commanded closed, a jaw within `success_strap_gap` (10 mm) of the strap, the bag still `lift_height` up (it only stays up if something holds it), AND joints 1-6 each within `final_pose_tol` (10 deg) of the `return_home` waypoint of `grasp_poses.json`. The episode **ends on first success** |
| **collision** | contact force on the **lower 95%** of the bag body (`collision_lower_frac`). The top rim and the handle are free |

The episode ends on first success, if the bag is pushed more than 35 cm from the tray centre, or at the timeout. The
timeout is set from the demo: demo length + `episode_extra_s` (4 s), so 30.9 s with the current pose file.

Bag spawn randomization follows a curriculum (`rand_curriculum`). The ranges (`bag_xy_range` ±5 cm, `bag_yaw_range`
±20°, `bag_tilt_range` 5°) are multiplied by a scale that starts at 0 and goes up by 0.1 each time a window of
~`num_envs` episodes reaches 80% final-pose rate. `env.rand_curriculum=false` uses the full ranges from the start.

**Demo check (zero actions, 16 envs).** Nominal bag pose: 94% final-pose rate, success at 25.3 s. Full randomization:
38%. Re-run after editing the pose file:

```bash
uv run python scripts/environments/zero_agent.py --task Isaac-BagLift-Direct --num_envs 16 --max_steps 1000 \
  env.rand_curriculum=true    # nominal pose; env.rand_curriculum=false for full randomization
```

## Deformable handle (`USE_DEFORMABLE_HANDLE` in `bag_sand_scene.py`)

`True` (default): the strap is a bendable PhysX **surface deformable** (a 1 mm thick, 13 mm wide shell, ~440 nodes),
clamped at its two feet to a body-only rigid bag. It is stiff enough to keep its arch under gravity and yields when the
jaws press it. `False`: the original fully rigid bag, nothing else changes (one-line revert).

What changes in deformable mode:

- **Grasp** is the geometric test in the table above. PhysX cannot report contacts on deformables, and a thin strap
  lets the jaws close almost all the way, so there is no grasp-force threshold.
- **Collision** = force on the lower 95% of the bag body. The top rim and the handle are free. The strap itself is
  not sensed.
- **Reset**: the strap's nodes are laid onto the bag's randomised pose at rest; the clamp is authored per env in
  `BagLiftEnv._setup_scene` (`OmniPhysicsVtxXformAttachment` + `OmniPhysicsElementCollisionFilter`).
- `grasp_pose_tool.py` prints each jaw's distance to the strap, how many strap nodes rest on the rigid jaw or sit
  between the jaws, and their depth in the mouth.

Tuning (`bag_sand_scene.py`, "BAG HANDLE (deformable)"): `HANDLE_STIFFNESS` (3e9 Pa, also sets the stretch/shear/bend overrides; raise if it sags, lower if
the jaws cannot dent it), `HANDLE_BEND_FACTOR`, `HANDLE_THICKNESS`, `HANDLE_DAMPING`, `HANDLE_SOLVER_ITERS`. Regenerate the strap mesh and jaw
points after editing the source USDs with `uv run python .../assets/build_handle_assets.py`.

## Rewards (`_get_rewards`)

Per-step terms are multiplied by `step_dt`; success is a flat one-off. Weights are in `bag_lift_env_cfg.py`.
reach, align and reached only pay **before** the grasp, so they do not pull the hand back down during the carry.

| Term | Weight | Meaning |
|---|---|---|
| reach | +0.5 | rigid tip toward the recorded goal tip position (`grasp_goal.json`, falls back to `pinch_target`); not while grasped |
| align | +0.5 | rigid jaw matching the recorded goal orientation (near the target); not while grasped |
| reached | +0.5 | rigid tip within `reach_radius` (15 mm) of the goal; not while grasped |
| close | +1.0 | gripper closed fraction, weighted by how close a jaw is to the strap nodes (`1 - tanh(gap / 1 cm)`) |
| track | +2.0 | all 7 joints toward the `grasp_poses.json` teacher pose this step (`1 - tanh(rms / 0.25)`). Zero residual replays the demo |
| grasp | +5.0 | grasp condition above is true (larger than hovering at the pinch without holding) |
| squeeze | +2.0 | gripper closed fraction while grasped |
| ee up | +3.0 | rigid-jaw long axis toward world +Z while grasped (point the EE up before carrying home) |
| return | +8.0 | joints 1-6 toward the `return_home` pose (`1 - tanh(rms error / 0.5)`), paid only while holding the strap |
| hand up | +5.0 | rigid-tip height above its resting pinch height, 0..5 cm, while grasped, faded out when the closest jaw is more than ~2 cm from the strap nodes (`carry_radius`), so lifting the hand off a slipping strap earns nothing |
| lift | +5.0 | bag lift progress 0..1 over 5 cm while grasped |
| **collision** | **-0.5** | `tanh(force / 5 N)` on the lower 95% of the bag body (top rim + handle free), **ramped in** from 0 over `collision_ramp_steps` |
| action rate | -0.01 | jerky residual |
| residual | -0.08 | `||action||^2`: prefer riding the teacher when the demo already works |
| time | -2.0 / s | living cost |
| success | +200 | flat one-off on first success. Larger than what staying grasped would still earn: the holding-gated terms total ~28/s and at most ~4 s remain after the demo (~110) |

Why the ramp: with the full penalty from step 0 the policy learned to stay away from the bag (collision rate
fell to 0, tip distance grew to 20-30 cm, grasp stayed 0). Watch `Metrics/collision_weight` in TensorBoard.

## Terminal output while training

One line is printed every time about `num_envs` episodes have finished (rates are over those episodes):

```
[BagLift] episodes:  128 | reach rate: 0.250 | grasp rate: 0.164 | lift (5 cm) rate: 0.031 | final-pose rate: 0.008 | collision rate: 0.742 | time to success: 24.1 s | bag randomization: 0.3
```

- `reach rate`: fraction of episodes where the rigid tip got within 15 mm of the pinch target at least once.
- `grasp rate`: fraction of episodes in which the grasp condition held at least once.
- `lift rate`: fraction of episodes in which the bag was lifted 5 cm while grasped at least once.
- `final-pose rate`: fraction of episodes that succeeded (see the task table). The terminal line also prints
  `at pose`, the fraction that hit the joint box even if the bag was not held.
- `collision rate`: fraction of episodes with any penalised bag contact.
- `time to success`: mean seconds from reset to first success, among the episodes in the window that succeeded.
- `bag randomization`: the curriculum scale on the spawn ranges (0 = nominal pose, 1 = full ranges).

The same values go to TensorBoard as `Metrics/reach_rate`, `Metrics/grasp_rate`, `Metrics/lift_rate`, `Metrics/final_pose_rate`, `Metrics/at_final_pose_rate`, `Metrics/collision_rate`, `Metrics/time_to_success_s`, `Metrics/episode_length_s`, `Metrics/rand_scale`
(plus `Metrics/grasped`, `Metrics/grasp_raw` (before debouncing), `Metrics/hand_rise`, `Metrics/ee_up`,
`Metrics/force_jaw_on_body`, `Metrics/force_other_links`, `Metrics/collision_weight`, `Metrics/tip_dist`,
`Metrics/collision_force`, `Metrics/lift_height`, `Metrics/approach_err_deg`, `Metrics/demo_track_err_deg`).

Watch `Metrics/final_pose_rate` together with `Metrics/rand_scale`: the rate dips each time the curriculum widens the
spawn ranges, then should recover.

**Control.** Joint targets are `teacher(t) + scale * action`. `scale` is `action_scale` (0.35 rad, ~20°) for most of the
demo and shrinks to `carry_action_scale` (0.07 rad, ~4°) over the last `residual_fade_frac` (25%) of it. It never
reaches zero, so the policy can still hold the strap on the way home, but 4° cannot push the arm out of the 10° box.
The teacher is also the speed ceiling: to go faster, raise `env.demo_speed` (1.2 = 20% faster teacher) or the
`speed_deg_per_s` values in `grasp_poses.json`.

**Observations (47).** Joint positions relative to home (7), joint velocities (7), joints minus the teacher pose (7),
demo phase (1), tip to goal in the world frame (3), bag position (3), bag quaternion with a canonical sign (4),
last actions (7), tip in the bag frame (3), and the grasp state (5): rigid and moving jaw-to-strap gaps (clamped at
10 cm), grasped, bag lift, gripper closed fraction. The strap gaps only exist in simulation, so this is a teacher
policy; a real-arm policy would need a student trained on camera views.

## Commands (from the Isaac Lab repo root)

```bash
# look at the scene (4 envs, Isaac Sim window); the smoke test needs a smaller minibatch
uv run --extra rl-games isaaclab train --rl_library rl_games \
  --task=Isaac-BagLift-Direct --num_envs 4 --viz kit --max_iterations 5 agent.params.config.minibatch_size=256

# train (headless is the default); logs go to logs/rl_games/bag_lift_v2/
uv run --extra rl-games isaaclab train --rl_library rl_games --task=Isaac-BagLift-Direct --num_envs 128

# watch a checkpoint
uv run --extra rl-games isaaclab play --rl_library rl_games \
  --task=Isaac-BagLift-Direct --num_envs 4 --checkpoint latest --viz kit

uv run tensorboard --logdir logs/rl_games
```

PPO (`agents/rl_games_direct_ppo_cfg.yaml`): `gamma` 0.997 and `tau` 0.97 (an ~11 s horizon at 30 Hz, so the success
bonus at the end of the demo is visible), `horizon_length` 64, `minibatch_size` 2048, `mini_epochs` 4, learning rate
2e-4. `--num_envs` must satisfy `(num_envs * 64) % 2048 == 0`, so any multiple of 32.
Any `BagLiftEnvCfg` value can be overridden with Hydra, e.g. `env.demo_speed=1.2 env.rew_scale_collision=-10.0`.

## Hand-posed grasp (no learning): `grasp_pose_tool.py`

Describe the grasp as joint angles in `grasp_poses.json` (created on first run with the home pose), then test it
open-loop. The bag is not randomised and there is no action noise, so results are repeatable.

```bash
# live: edit and save grasp_poses.json, the arm in the Kit window follows; the terminal prints the tip pose in the
# bag frame, jaw-to-strap gaps, strap node counts, and whether grasp is true
uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit

# run: pregrasp -> grasp -> grasp_closed -> lift -> ee_up -> return_home, then report whether the bag rose 5 cm
# and whether the env's success test fired. Per stage it prints the debounced and raw grasp rate. Without
# --viz kit it exits when the sequence ends.
uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit --mode run

# run only up to grasp_closed and save every control step (joints, tip in the bag frame, jaw-to-strap gaps, grasped)
# to grasp_trajectory.json
uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit --mode run \
    --stop_after grasp_closed --save_traj
```

Trajectory timing lives in the same file. `"speed_deg_per_s"` is `{waypoint: {joint: deg/s}}`: each joint moves into
that waypoint at its own speed (duration = |change| / speed, smooth start and stop, peak 1.5x) and waits for the
slowest one; `"default"` covers unlisted joints and a plain number applies to every joint. The motors cap at
28.6 deg/s, so keep speeds at or below 19 deg/s (peak 28.5). Each waypoint is followed by a `demo_hold_s` (0.3 s)
pause. An optional `"sequence"` list lets you add your own waypoints.

Current timing: approach moves (`pregrasp`, `grasp`, `grasp_closed`) at 19 deg/s; the carry (`lift`, `ee_up`) at
12 deg/s and `return_home` at 15 deg/s. With the carry at 19 deg/s the strap stored tension during `lift` and snapped
out of the jaws (gap 4 mm to 68 mm in one step) before the bag was up. The wrist pitch (`ee_up`) comes after `lift`,
once the bag hangs from the jaws.

The tool disables the demo residual (`use_demo_residual = False`) and widens `action_scale` to 4 rad so a single
command can jump to any pose inside the USD joint limits. Training sets `target = teacher(t) + 0.35 rad * action`,
so a zero action replays `grasp_poses.json` and the policy only learns corrections for bag motion.

**Record the target for training.** Record at the open-jaw `grasp` waypoint, where the tip should be before closing:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --record --show grasp
```

This plays the sequence up to that waypoint (so the tip ends up where the demo really puts it) and writes
`grasp_goal.json`: rigid-jaw tip position and orientation in the bag frame, gripper angle and joints. Re-record after
editing the approach waypoints. When that file exists the env uses the recorded tip position as the reach target and
rewards matching the recorded jaw orientation; without it the env falls back to `pinch_target` and a "jaw points
down" alignment.

## Files

```
bag_lift/
├── __init__.py                          registers Isaac-BagLift-Direct
├── bag_sand_scene.py                    arm, bag, tray, contact sensor, light (physical constants)
├── bag_lift_env_cfg.py                  every number: rewards, success thresholds, randomization
├── bag_lift_env.py                      logic: task state, rewards, dones, reset, metrics
├── agents/rl_games_direct_ppo_cfg.yaml  PPO + MLP [256, 128]
└── assets/                              USD wrappers (arm, bag, bag body only, strap ribbon, tray), the generated
                                         bag_handle_info.json / jaw_points.json, and the build scripts
```

## Notes

- Bag is one rigid body, `BAG_MASS` = 0.08 kg (was 0.2; the strap popped through the jaw under that load). The
  gripper motor is capped at `GRIPPER_EFFORT_LIMIT` = 0.8 N*m instead of the 3 N*m of the other joints. With the
  deformable handle the body is the rigid bag and the strap is the deformable. With `USE_DEFORMABLE_HANDLE=False`,
  grasp falls back to both jaws contacting the strap by location (`handle_zone_*`), still with no force threshold.
- **Handle geometry.** The handle is a thin strap (13 mm wide, about 1 mm thick) bent into a U that sticks 38 mm out
  of the bag body, not a hole to thread. The rigid jaw is a wedge (9 mm at the tip, 25 mm at 5 cm, then a 49 mm thick
  housing), so a 5 cm "through the loop" target was unreachable (run of 5000 epochs: in-loop 0.85, grasp 0.000, tip
  stuck at the loop plane). The task is now a pinch of the strap with one jaw on each side.
- **Handle collider.** `Rigid Bag.usd` gives the handle a `convexHull` collider, and the hull of an arch is a solid
  block. `BAG_CFG` cooks the handle as a convex decomposition so the U is open. Check in Kit: Window > Debug > Physics >
  Collision > show collision shapes.
- Gripper is policy-controlled from step 0. The collision penalty is what teaches it to keep clear of the bag body.
- Absolute USD paths (`/home/joshua/Documents/...`, `/home/joshua/Downloads/...`) are in `bag_sand_scene.py` and
  `assets/*.usda`.
- `--viz kit` opens the window; red sphere = rigid tip, green = pinch target on the strap.
