# Generated reset states for sparse cube pickup

This reference example exercises `isaaclab.envs.reset.ResetStateBuffer`: task callbacks
generate candidate states through simulation, reject invalid candidates, and freeze the
accepted states before reinforcement learning. The example currently supports Newton MJWarp.

Support owner during reset API development: Maximilian Krause. Evaluate this example
with the reset-state-buffer API before promotion to a stable reference. Remove or
relocate it if that API is withdrawn or another example covers the same capability.

A Franka carries a grasped 40 mm cube within 0.04 m of the fixed target point
`(0.5, 0.0, 0.15) m` in the robot root frame. Success is immediate when the cube
reaches the point while grasped. The only reward is a terminal `+1` for that success.
There are no reaching, grasping, motion, or progress rewards. Relative Cartesian IK and
a binary gripper command use the existing Franka asset and action implementations.
The gripper opens for actions above 0.5 and closes otherwise.

The buffer contains four stages from physical scripted rollouts: home with the cube on
the table, approach targets interpolated from home to 0.10 m above the cube, open-finger
pregrasps spanning 0–0.10 m above it, and established grasps lifted by 0.025–0.098 m.
These continuous starts connect the pickup path without already satisfying the goal.
The generator closes the real fingers and performs a short lift; it
never places a cube into a closed hand. A grasp must survive state restoration and
continued physics before acceptance. Collection has a fixed attempt limit and reports
acceptance counts. Snapshots include velocities in environment-relative coordinates.
Acceptance excludes the same grasped-cube goal predicate used by reward and termination.
The goal is observed through the existing pose command. Replay preserves
physical velocities while clearing velocity targets for the position-controlled arm.

Training records completed-episode outcomes against the previous reset row before
clearing task state. Sampling targets rows with approximately 50% recent success, with
uniform coverage retained. The policy observes neither reset-row IDs nor teacher actions.
The buffer changes the initial-state distribution; it supplies no demonstration loss.

```bash
# Prove bank generation, partial replay isolation, and finite stepping.
uv run python scripts/environments/cube_pickup.py --mode generate \
  --num_envs 32 --bank_size 128 --seed 0 --output /tmp/pickup-generation.json

# Prove pickup from ordinary home/table starts with the scripted controller.
uv run python scripts/environments/cube_pickup.py --mode scripted \
  --num_envs 32 --episodes 64 --seed 10000 --output /tmp/pickup-scripted.json

# Train; the actual generated bank and collection report are saved in the run directory.
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Cube-Pickup-Franka \
  --num_envs 1024 --max_iterations 3000 --seed 0

# Evaluate a checkpoint on full pickup attempts with a held-out seed and no reset bank.
uv run python scripts/environments/cube_pickup.py --mode evaluate \
  --checkpoint /path/to/model.pt --num_envs 64 --episodes 512 --seed 10000 \
  --output /tmp/pickup-evaluation.json
```

Set `env.reset_mode=uniform` when training to compare fixed uniform bank sampling, or
`env.reset_mode=table` to train from ordinary table starts without a bank. Change
`env.reset_buffer.size` to control collection size and `env.reset_buffer.seed` to vary
the generated bank independently of the training seed.

Training success on the reset mixture does not establish full pickup learning. The
evaluation runner always resets the arm home and the cube onto the table, reports
completed episodes and binary successes, and bounds the rollout by `--steps`. Evaluate
multiple held-out seeds before claiming generalization.
