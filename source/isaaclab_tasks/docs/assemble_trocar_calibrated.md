# Calibrated G1 trocar assembly

The existing `IsaacContrib-Assemble-Trocar-G129-Dex3` and `-Eval` tasks retain their remote assets,
224 x 224 camera presets, initial pose, reward defaults, 50 Hz control and existing RLinf recipe.

The opt-in `IsaacContrib-Assemble-Trocar-G129-Dex3-Calibrated` and `-Calibrated-Eval` tasks use the
rl2real camera calibration and packaged scene. Their reusable configurations are
`CalibratedAssembleTrocarSceneCfg`, `G1AssembleTrocarCalibratedEnvCfg` and
`G1AssembleTrocarCalibratedEvalEnvCfg` in `contrib/assemble_trocar/calibrated_env_cfg.py`.
They provide raw RGB and do not depend on Cosmos or a model service.

Extract `assemble-trocar-assets.zip` and set `ASSET_LOCAL` before launching Isaac Lab to the directory
containing `Robots/` and `Props/`. The default is `/workspace/assets`; an HTTPS prefix is also accepted.
This asset layout uses `g1_29dof_with_dex3_base_fix02.usd`, `scene03cell.usd`, `Trocar004_nodeform.usd`,
`DisposableLaparoscopicPunctureDevice006.usd` and `SurgicalTray001_FixedBoxProp.usd`.
The asset pack must be supplied separately; this change does not redistribute it.

The calibrated task captures 320 x 240 RGB from the front and both wrists and controls the robot at
30 Hz (240 Hz physics, decimation 8). Its 0.08 m lift criterion uses the first instrument's center of
mass because its USD root is offset. The optional `use_com_for_lift` reward argument defaults to
False, preserving the original instrument-root criterion for existing callers.

For GR00T, select the separate `isaaclab_ppo_gr00t_assemble_trocar_calibrated` RLinf recipe. Its image
adapter crops and resizes the camera output to the policy's 224 x 224 input; the original recipe is
unchanged. Checkpoint instruction and absolute joint-position action conversion must match the policy.
The successfully evaluated cotrain combination is `cosmos50/checkpoints-93` with
`install trocar from tray`, absolute joint positions, 40 reset warmup steps and 512 evaluation steps.
Set the instruction in both `env.train` and `env.eval`, under `init_params` and `isaaclab`, in a copied
recipe. These settings do not imply that another checkpoint has the same insertion performance.

Cosmos Transfer is composed by the separate `isaaclab_demos` assemble-trocar demo. Shared model
clients and services, prompts, the three-camera grid, foreground compositing and PPO model residency
are outside Isaac Lab. External tasks are discovered in spawned RLinf simulators through the
`isaaclab.tasks` entry points. An external Gym task may register an `rlinf_runner_context` import
reference to manage optional resources around worker initialization and training.
