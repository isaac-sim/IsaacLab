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
weights allow the robot's legs to adapt to uneven ground. The play configuration
uses rough terrain and retains particle-material randomization.
