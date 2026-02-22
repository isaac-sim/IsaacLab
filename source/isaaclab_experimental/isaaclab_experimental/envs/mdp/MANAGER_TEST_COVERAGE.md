# Manager Implementation Test Coverage

> Which tasks from `run_single_manager_warp_sweep.sh` are needed to exercise every
> reachable code path in the warp manager implementations?

---

## Minimal Test Set

**4 tasks (env-ids `1,2,6,3`) cover every manager code path that existing tasks can reach.**

```bash
# Sweep command example
./run_single_manager_warp_sweep.sh default=0 target=2 env-ids=1,2,6,3
```

| env-id | Gym ID | Role |
|:------:|--------|------|
| 1 | `Isaac-Cartpole-Warp-v0` | Simplest baseline; `JointEffortAction`; `corruption=False`; no commands |
| 2 | `Isaac-Humanoid-Warp-v0` | Obs `scale`; per-joint action scale dict; class-based rewards |
| 6 | `Isaac-Velocity-Flat-Anymal-C-Warp-v0` | All 3 event modes; obs `noise`; sensor deps; velocity commands; terrain curriculum |
| 3 | `Isaac-Reach-Franka-Warp-v0` | Pose commands; `modify_reward_weight` curriculum |

### Why every other task is redundant

| Dropped | Reason |
|---------|--------|
| Ant (0) | Strict subset of Humanoid: fewer obs `scale` terms, no per-joint action dict, subset of class-based rewards |
| Velocity quadrupeds (5,7-8,12-14) | Identical manager structure to Anymal-C; differ only in hyperparameters (action scale, body names, joint names) |
| Velocity bipeds (9-11) | Add more reward/termination terms of the same types; no new manager code paths |
| Velocity rough (15) | Same manager structure as flat; terrain config is scene-level, not manager-level |
| Reach-UR10 (4) | Identical manager structure to Reach-Franka; differs only in robot asset and body names |

---

## Manager Code Path Coverage Matrix

### Observation Manager

Source: `isaaclab_experimental/managers/observation_manager.py`

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| Basic term compute (`func(env, out)`) | `compute_group` L494 | Y | Y | Y | Y | YES |
| `scale` kernel (`_apply_scale`) | `compute_group` L517-523 | | Y | | | YES |
| `clip` kernel (`_apply_clip`) | `compute_group` L510-516 | | | | | **NO** |
| `noise` application | `compute_group` L503-508 | | | Y | Y | YES |
| `modifiers` pipeline | `compute_group` L498-501 | | | | | **NO** |
| `enable_corruption=False` (skip noise) | `compute_group` L496 | Y | Y | | | YES |
| `enable_corruption=True` | `compute_group` L496 | | | Y | Y | YES |
| `concatenate_terms=True` (contiguous buf) | `_prepare_terms` L637 | Y | Y | Y | Y | YES |
| `concatenate_terms=False` (separate bufs) | `_prepare_terms` L653 | | | | | **NO** |
| Dim inference: `axes` (root-state obs) | `_infer_term_dim_scalar` | | Y | Y | | YES |
| Dim inference: `out_dim` int (custom obs) | `_infer_term_dim_scalar` | | Y | | | YES |
| Dim inference: `"joint"` sentinel | `_infer_term_dim_scalar` | Y | Y | Y | Y | YES |
| Dim inference: `"command"` sentinel | `_infer_term_dim_scalar` | | | Y | Y | YES |
| Dim inference: `"action"` sentinel | `_infer_term_dim_scalar` | | Y | Y | Y | YES |
| Dim inference: `"body:N"` sentinel | `_infer_term_dim_scalar` | | | | | **NO** |
| Cross-manager obs (commands) | `generated_commands` | | | Y | Y | YES |
| Cross-manager obs (last_action) | `last_action` | | Y | Y | Y | YES |
| Class-based modifier `.reset()` | `reset` L402-404 | | | | | **NO** |

### Action Manager

Source: `isaaclab_experimental/managers/action_manager.py`

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| Single-term action processing | `process_action` L410-431 | Y | Y | Y | Y | YES |
| Multi-term offset splitting | `process_action` L425-427 | | | | | **NO** |
| `prev_action` copy kernel | `process_action` L421-422 | Y | Y | Y | Y | YES |
| `JointPositionAction` | `joint_actions.py` | | | Y | Y | YES |
| `JointEffortAction` | `joint_actions.py` | Y | Y | | | YES |
| `RelativeJointPositionAction` | `joint_actions.py` | | | | | **NO** |
| `JointVelocityAction` | `joint_actions.py` | | | | | **NO** |
| `BinaryJointPositionAction` | `binary_joint_actions.py` | | | | | **NO** |
| `BinaryJointVelocityAction` | `binary_joint_actions.py` | | | | | **NO** |
| `JointPositionToLimitsAction` | `joint_actions_to_limits.py` | | | | | **NO** |
| `EMAJointPositionToLimitsAction` | `joint_actions_to_limits.py` | | | | | **NO** |
| `NonHolonomicAction` | `non_holonomic_actions.py` | | | | | **NO** |
| Per-joint scale dict | `JointEffortAction.__init__` | | Y | | | YES |
| `use_default_offset=True` | `JointPositionAction.__init__` | | | Y | | YES |

### Event Manager

Source: `isaaclab_experimental/managers/event_manager.py`

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| `startup` mode dispatch | `_apply_startup` | | | Y | | YES |
| `reset` mode dispatch | `_apply_reset` L369-388 | Y | Y | Y | Y | YES |
| `interval` mode (per-env timer) | `_apply_interval` L353-367 | | | Y | | YES |
| `interval` mode (global timer) | `_apply_interval` L336-352 | | | | | **NO** |
| `_interval_step_per_env` kernel | L65-82 | | | Y | | YES |
| `_interval_step_global` kernel | L86-102 | | | | | **NO** |
| `_interval_reset_selected` (re-sample on env reset) | `reset` L262-278 | | | Y | | YES |
| `min_step_count_between_reset` logic | `_reset_compute_valid_mask` L128-158 | | | | | **NO** |
| Class-based event terms | `_prepare_terms` | | | | | **NO** |
| Function-based event terms | `_prepare_terms` | Y | Y | Y | Y | YES |

### Reward Manager

Source: `isaaclab_experimental/managers/reward_manager.py`

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| Function-based rewards | `compute` | Y | Y | Y | Y | YES |
| Class-based rewards (init/reset/call) | `compute`, `reset` | | Y | | | YES |
| `_reward_finalize` kernel (weighted sum) | `compute` | Y | Y | Y | Y | YES |
| `_reward_pre_compute_reset` (zero per step) | `compute` | Y | Y | Y | Y | YES |
| Episode sum tracking + reset logging | `reset` | Y | Y | Y | Y | YES |
| Sensor-dependent rewards (`wp.from_torch`) | via `undesired_contacts` etc. | | | Y | | YES |
| Command-dependent rewards (`wp.from_torch`) | via `track_lin_vel_xy_exp` etc. | | | Y | Y | YES |

### Termination Manager

Source: `isaaclab_experimental/managers/termination_manager.py`

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| `_termination_finalize` (reduce to dones) | `compute` | Y | Y | Y | Y | YES |
| `time_out=True` flag handling | `_term_is_time_out_wp` | Y | Y | Y | Y | YES |
| `time_out=False` (real termination) | `_termination_finalize` | Y | Y | Y | | YES |
| Sensor-based termination | via `illegal_contact` | | | Y | | YES |
| Reset mean logging | `_termination_reset_mean_all_2d` | Y | Y | Y | Y | YES |

### Command Manager

Source: `isaaclab_experimental/managers/command_manager.py`

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| No commands (empty manager) | `__init__` | Y | Y | | | YES |
| `UniformVelocityCommand` | `command_manager.py` | | | Y | | YES |
| `UniformPoseCommand` | `command_manager.py` | | | | Y | YES |
| Resample on reset | `_step_time_left_and_build_resample_mask` | | | Y | Y | YES |

### Curriculum Manager

Note: Curriculum manager is not warp-migrated (runs at reset, not per-step). Included for completeness.

| Code Path | Location | Cartpole | Humanoid | Velocity | Reach | Covered? |
|-----------|----------|:--------:|:--------:|:--------:|:-----:|:--------:|
| No curriculum (empty manager) | | Y | Y | | | YES |
| Custom curriculum (`terrain_levels_vel`) | torch-based, reset-only | | | Y | | YES |
| `modify_reward_weight` (stable) | forwarded from stable | | | | Y | YES |

---

## Uncovered Manager Code Paths

These code paths exist in the manager implementations but **no existing migrated task exercises them**.

### Observation Manager (5 gaps)

| Gap | Code Location | What triggers it |
|-----|---------------|------------------|
| `clip` kernel | `_apply_clip` L71-74, launched at L510-516 | Any `ObsTermCfg` with `clip=(lo, hi)` |
| `modifiers` pipeline | L498-501 compute, L402-404 reset | Any `ObsTermCfg` with `modifiers=[...]` |
| `concatenate_terms=False` | `_prepare_terms` L653+ | An obs group with `concatenate_terms=False` |
| `scale` as tuple | `_prepare_terms` L691-705 | `ObsTermCfg(scale=(1.0, 2.0, ...))` with per-element values |
| `out_dim="body:N"` inference | `_infer_term_dim_scalar` | An obs using `body_pose_w` or `body_projected_gravity_b` |

### Action Manager (8 gaps)

| Gap | Code Location | What triggers it |
|-----|---------------|------------------|
| Multi-term offset splitting | `process_action` L425-427 | A task with 2+ non-None action terms |
| `RelativeJointPositionAction` | `joint_actions.py` | `RelativeJointPositionActionCfg` |
| `JointVelocityAction` | `joint_actions.py` | `JointVelocityActionCfg` |
| `BinaryJointPositionAction` | `binary_joint_actions.py` | `BinaryJointPositionActionCfg` |
| `BinaryJointVelocityAction` | `binary_joint_actions.py` | `BinaryJointVelocityActionCfg` |
| `JointPositionToLimitsAction` | `joint_actions_to_limits.py` | `JointPositionToLimitsActionCfg` |
| `EMAJointPositionToLimitsAction` | `joint_actions_to_limits.py` | `EMAJointPositionToLimitsActionCfg` |
| `NonHolonomicAction` | `non_holonomic_actions.py` | `NonHolonomicActionCfg` |

### Event Manager (3 gaps)

| Gap | Code Location | What triggers it |
|-----|---------------|------------------|
| `interval` with `is_global_time=True` | `_apply_interval` L336-352, `_interval_step_global` L86-102 | Any `EventTermCfg(mode="interval", is_global_time=True)` |
| `min_step_count_between_reset` | `_reset_compute_valid_mask` L128-158 | Any `EventTermCfg(mode="reset", min_step_count_between_reset=N)` where N > 0 |
| Class-based event terms | `_prepare_terms` class instantiation path | `randomize_rigid_body_material`, `randomize_rigid_body_mass`, `randomize_actuator_gains`, `randomize_joint_parameters` |

---

## Coverage Summary

| Manager | Total Paths | Covered | Uncovered |
|---------|:-----------:|:-------:|:---------:|
| Observation | 18 | 13 | **5** |
| Action | 14 | 6 | **8** |
| Event | 10 | 7 | **3** |
| Reward | 7 | 7 | 0 |
| Termination | 5 | 5 | 0 |
| Command | 4 | 4 | 0 |
| Curriculum | 3 | 3 | 0 |
| **Total** | **61** | **45 (74%)** | **16 (26%)** |

The 16 uncovered paths break down as:
- **8 action term types** — no migrated task uses these action classes
- **5 obs post-processing features** — `clip`, `modifiers`, tuple `scale`, `concatenate_terms=False`, `body:N` dim inference
- **3 event features** — global interval, reset rate-limiting, class-based randomization events

---

## Repo-Wide Search for Gap Coverage

Searched the entire IsaacLab repo (stable tasks, direct-workflow tasks, test configs,
examples) for any usage of the 16 uncovered features.

### Already covered by existing unit tests (not end-to-end)

These features have **no task config usage** anywhere in the repo but are exercised by
dedicated unit tests. They do NOT need new task coverage — the unit tests validate the
warp implementation in isolation.

| Gap | Unit Test | File |
|-----|-----------|------|
| `JointVelocityAction` | `TestJointActions` | `isaaclab_experimental/test/envs/mdp/test_action_warp_parity.py:231` |
| `BinaryJointPositionAction` | `TestBinaryJointActions` | `test_action_warp_parity.py:300` |
| `BinaryJointVelocityAction` | `TestBinaryJointActions` | `test_action_warp_parity.py:300` |
| `JointPositionToLimitsAction` | `TestJointPositionToLimitsActions` | `test_action_warp_parity.py:374` |
| `EMAJointPositionToLimitsAction` | `TestJointPositionToLimitsActions` | `test_action_warp_parity.py:374` |
| `NonHolonomicAction` | `TestNonHolonomicAction` | `test_action_warp_parity.py:448` |
| `body_pose_w` (`out_dim="body:N"`) | `test_body_pose_w` | `test_mdp_warp_parity_new_terms.py:445,815` |
| `body_projected_gravity_b` | `test_body_projected_gravity_b` | `test_mdp_warp_parity_new_terms.py:435,803` |

### Closable by uncommenting existing config (1 gap)

The velocity env config has a complete, ready-to-use `randomize_rigid_body_material`
event term that is commented out. Uncommenting it would close the class-based event gap.

```
# source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py  L184-194
# physics_material = EventTerm(
#    func=mdp.randomize_rigid_body_material,
#    mode="startup",
#    params={
#        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#        "static_friction_range": (0.8, 0.8),
#        "dynamic_friction_range": (0.6, 0.6),
#        "restitution_range": (0.0, 0.0),
#        "num_buckets": 64,
#    },
# )
```

Note: `randomize_rigid_body_mass` is also commented out but marked as causing NaNs.

| Gap | Stable Config Location | Status |
|-----|----------------------|--------|
| Class-based events (`randomize_rigid_body_material`) | `velocity_env_cfg.py:184-194` | Complete config, degenerate ranges (no actual randomization) — safe to uncomment |
| Class-based events (`randomize_rigid_body_mass`) | `velocity_env_cfg.py:196-205` | Known broken (NaN) — do NOT uncomment |

### Used only in Dexsuite — complex, not practical to migrate for coverage alone (2 gaps)

| Gap | Stable Task | Why impractical |
|-----|------------|-----------------|
| `clip` on ObsTerm | Dexsuite (`dexsuite_env_cfg.py:157,176`) | 15+ custom MDP terms, ADR curriculum, multi-obs-group — not migrated |
| `RelativeJointPositionAction` | Dexsuite Kuka-Allegro (`dexsuite_kuka_allegro_env_cfg.py:40`) | Same complexity |

### True dead code — zero usage anywhere in the repo (5 gaps)

These manager code paths are implemented but have **no usage in any task config, test,
or example** across the entire codebase. They are forward-looking infrastructure.

| Gap | Description |
|-----|-------------|
| `modifiers` pipeline | `ObsTermCfg.modifiers` — defined but never set in any config |
| `scale` as tuple | Per-element varying scale — only float `scale=` is ever used |
| `concatenate_terms=False` | Every obs group in every config sets `True` |
| `is_global_time=True` | Tested in stable unit test only (`test_event_manager.py:276`); no task config |
| `min_step_count_between_reset` | Tested in stable unit test only (`test_event_manager.py:331`); no task config |
| Multi-term action splitting | Reach declares `gripper_action` slot but never populates it |

---

## Revised Gap Classification

| Category | Count | Gaps |
|----------|:-----:|------|
| Covered by unit tests (no task needed) | 8 | 6 action types + 2 body obs (`body:N` inference) |
| Closable by uncommenting stable config | 1 | Class-based events (`randomize_rigid_body_material`) |
| Blocked behind complex task migration | 2 | `clip`, `RelativeJointPositionAction` (Dexsuite) |
| True dead code (no usage anywhere) | 5 | `modifiers`, tuple `scale`, `concatenate_terms=False`, `is_global_time`, `min_step_count_between_reset` |
| **Total** | **16** | |

### Effective end-to-end gap after accounting for unit tests: 8

Of those 8:
- **1 is actionable now** (uncomment DR event in velocity config)
- **2 require Dexsuite migration** (large effort, low priority)
- **5 have zero usage anywhere** (cannot be tested without writing new configs)

---

## Open: Per-MDP Capturability Tracking

### Problem

Manager mode=2 (WARP_CAPTURED) assumes all MDP terms are CUDA-graph-capturable.
Some terms call non-capturable external APIs (e.g., `write_root_pose_to_sim`,
`set_external_force_and_torque`). If any term is non-capturable, the manager
should fall back to mode=1.

Not capturability issues:
- `wp.from_torch` — stable pointers, fine in graphs
- `wp.zeros` in `hasattr` guards — solvable via warmup in `_wp_capture_or_launch`

### Proposed: `@warp_capturable` decorator

By default all MDP terms are assumed capturable (True). Only non-capturable
terms need `@warp_capturable(False)`. The decorator sets an attribute directly
on the function (no wrapper), so it composes safely with any other decorator
in any order.

```python
def warp_capturable(capturable: bool):
    """Annotate an MDP term's CUDA-graph capturability. Default assumption: True."""
    def decorator(func):
        func._warp_capturable = capturable
        return func  # no wrapper
    return decorator

def is_warp_capturable(func) -> bool:
    """Check capturability. Default: True. Checks __wrapped__ for decorated fns."""
    for f in (func, getattr(func, '__wrapped__', None)):
        if f is not None:
            val = getattr(f, '_warp_capturable', None)
            if val is not None:
                return val
    return True
```

Usage:
```python
@warp_capturable(False)
def apply_external_force_torque(env, env_mask, ...):
    ...
```

Manager integration: during `_prepare_terms`, check all terms. If any returns
`is_warp_capturable(func) == False`, fall back to mode=1 with a warning.

### Terms requiring `@warp_capturable(False)`

| Term | Non-capturable dependency |
|------|--------------------------|
| `apply_external_force_torque` | `wrench_composer.set_forces_and_torques()` |
| `reset_root_state_uniform` | `write_root_pose_to_sim()` / `write_root_velocity_to_sim()` |
| `reset_root_state_with_random_orientation` | Same |
| `reset_root_state_from_terrain` | Same |
| `reset_scene_to_default` | Same |
| `push_by_setting_velocity` | `write_root_velocity_to_sim()` |
