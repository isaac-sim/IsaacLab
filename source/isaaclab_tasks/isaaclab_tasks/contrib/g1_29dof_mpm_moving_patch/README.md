# G1 locomotion with a moving MPM particle patch

This manager-based task keeps a finite sand particle patch around a G1 robot while it traverses generated terrain. MJWarp simulates the robot. Implicit MPM simulates sand particles, and proxy coupling transfers sand reaction forces to the robot.

The moving particle patch contains simulated sand and boundary particles. Its center follows the robot on a discretized grid. Particles leaving the patch are recycled using heights queried from the fixed background terrain mesh. The rigid terrain collider beneath the sand is lowered by the configured sand depth.

Reset events sample particle material parameters in uniform or log-uniform space. Recycled particle masses remain consistent with each environment's sampled density. Observation/action symmetry uses the shared contributed API with task-local G1 mappings. OpenGL and RTX visualizers display the moving sand particles.

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch --num_envs 2
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-Play --num_envs 1 --wandb_run <run-id> --viz newton_rtx
```

Task settings are in `mpm_env_cfg.py`. Particle dimensions, materials, and terrain configuration are in `env_cfg/scene_cfg.py`; sampling ranges are in `env_cfg/event_cfg.py`. This initial task uses proxy coupling. Mixed rigid/MPM environments are introduced separately.

The robot asset is provided by `isaaclab_contrib.assets`. Ground texture assets are not included; the terrain uses its configured plain color. Coupled material updates require Newton's particle-material synchronization fix. Multi-environment MPM requires the Warp multi-environment cell-lookup fix; see the draft PR validation notes for tested dependency versions.

## Mixed rigid and MPM terrain

The mixed task uses one shared physics configuration and terrain generator. For an even number of environments, the first half receive MPM particles and terrain columns whose rigid supporting mesh is lowered by the sand depth. The second half contain no MPM particles and use rigid terrain columns at the generated surface height.

The default generator uses rough terrain. `PairedTerrainGenerator` generates the configured grid once and duplicates its geometry, origins, and sampled flat patches for the other contact group. `num_cols` is the column count **per group**, including odd values. The two grids meet at world Y=0. Other terrain generators can be selected through `env_cfg/mixed_scene_cfg.py`. The mixed environment combines robot foot reaction forces from the MPM coupler and MJWarp rigid contacts. Material randomization is restricted to the MPM environments.

`mpm_contact_margin` and `rigid_contact_margin` in `mixed_env_cfg.py` set fixed robot foot collider margins for the two environment groups. Rigid terrain uses zero foot collider margin by default. These values are not sampled during reset.

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain --num_envs 4
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain-Play --num_envs 4 --wandb_run <run-id> --viz newton_rtx
```

## Fine-tuning into a separate project

`--wandb_project` selects the source checkpoint project. `--log_project_name`
and `--experiment_name` select the destination project and local experiment folder.

```bash
uv run --no-sync --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch \
  --num_envs 64 --wandb_run <run-id> --wandb_project <source-project> \
  --logger wandb --log_project_name g1_29dof_mpm_rough_ppo \
  --experiment_name g1_29dof_mpm_rough_ppo
```

Standing contact rewards penalize unsupported feet while reducing the penalty
as base linear velocity, angular velocity, or tilt increases. Standing posture
weights allow the robot's legs to adapt to uneven ground. The MPM-only play configuration
uses rough terrain and retains particle-material randomization. Mixed-terrain play
uses difficult rough terrain with a 0.35 m sand layer and disables material randomization.
