# Isaac Lab PhysicsNeMo

This extension connects Isaac Lab Newton environments to
[PhysicsNeMo Robotics](https://github.com/NVIDIA/physicsnemo-robotics) learned dynamics. It
records environment episodes as PhysicsNeMo trajectory datasets, trains a control-conditioned
Neural Robot Dynamics (NeRD) model on them, and runs the resulting dynamics artifact as the Newton
solver of the same environment through an ordinary `NewtonCfg`.

```text
teacher env (MuJoCo Warp) ──record──▶ Zarr dataset ──train──▶ DynamicsArtifact ──deploy──▶ learned env
        TrajectoryDatasetRecorder     train_dynamics_artifact      PhysicsNeMoSolverCfg
```

## Architecture

The extension owns the Isaac Lab integration and uses PhysicsNeMo Robotics only as a learned-dynamics
library. It does not use the `physicsnemo_robotics.isaaclab` adapter, and it changes neither project.

| Layer | Owner | Contents |
| --- | --- | --- |
| Isaac Lab integration | this extension | `physics.PhysicsNeMoSolverCfg`, `physics.NewtonPhysicsNeMoManager`, state ownership validation, `NewtonBinding`, `preflight_learned_physics`, recording, training recipe, task configurations |
| Learned dynamics | PhysicsNeMo Robotics | `DynamicsArtifact`, `LearnedDynamicsSolver`, state codecs, feature encoders, trajectory schemas and datasets, NeRD models, training harness |

`NewtonPhysicsNeMoManager` follows the Isaac Lab
[Newton solver extension contract](../../../docs/source/developer-tools/extending_newton_solvers.rst):
`PhysicsNeMoSolverCfg.class_type` selects it through `NewtonCfg.solver_cfg`, and its `_build_solver`
publishes the learned solver with single-state stepping, no external body forces, and a collision
pipeline only when the artifact consumes contacts. Before the first step it checks the artifact
frame duration, the canonical frame, the solver-independent model identity, that the codec owns
every dynamic Newton state row, and that joint efforts carry every actuation command. Environment
resets reach the solver through the inherited reset hook, CUDA graph capture is disabled, contact
sensors are rejected, and a step with an external body wrench fails on the device.

## Installation

PhysicsNeMo Robotics is not published on a package index and pins older Newton, Warp, and Torch
releases than Isaac Lab. Install it into the Isaac Lab environment without its dependency pins, then
add its remaining dependencies. Isaac Lab keeps its own Newton, Warp, and Torch versions.

```bash
export ROBOTICS_ROOT=/absolute/path/to/physicsnemo-robotics

uv sync --frozen
uv pip install --python .venv/bin/python --no-deps -e "$ROBOTICS_ROOT"
uv pip install --python .venv/bin/python \
  "nvidia-physicsnemo==2.1.1" "jaxtyping>=0.3.3" "tensordict>=0.11,<0.12" "zarr>=3,<4"
uv pip install --python .venv/bin/python --no-deps -e source/isaaclab_physicsnemo
```

Run Python from `.venv/bin/python`, or use `uv run --no-sync`. A later `uv sync` restores the
locked environment and removes these packages.

The integration was validated with PhysicsNeMo Robotics `mataei/icra` at `4e4be69`, Newton 1.6.1,
Warp 1.17.0, Torch 2.12, and PhysicsNeMo 2.1.1 on one CUDA GPU.

## Examples

Each task has record, train, and deploy scripts under [`examples/`](../examples). Run them from the
repository root.

### Cartpole

```bash
python source/isaaclab_physicsnemo/examples/cartpole/record.py --output /tmp/cartpole-data
python source/isaaclab_physicsnemo/examples/cartpole/train.py \
  --dataset /tmp/cartpole-data --output /tmp/cartpole-train
python source/isaaclab_physicsnemo/examples/cartpole/deploy.py --artifact /tmp/cartpole-train/artifact
```

The Cartpole recording follows the task-independent data of NeRD and FlashNeRD
(`tasks.CartpoleExplorationEnvCfg`): episodes start anywhere on the track, at every pole angle, and with
large velocities. The model therefore deploys to the standard Cartpole task, whose states it covers.

`train_policy.py` is the downstream test. It trains the Isaac Lab Cartpole PPO policy either with MuJoCo
Warp or in the learned environment, then evaluates the policy with MuJoCo Warp:

```bash
python source/isaaclab_physicsnemo/examples/cartpole/train_policy.py --output /tmp/ppo-mjwarp
python source/isaaclab_physicsnemo/examples/cartpole/train_policy.py --output /tmp/ppo-learned \
  --artifact /tmp/cartpole-train/artifact
```

### ANYmal-D on flat ground

```bash
python source/isaaclab_physicsnemo/examples/anymal_d/record.py --output /tmp/anymal-data
python source/isaaclab_physicsnemo/examples/anymal_d/train.py \
  --dataset /tmp/anymal-data --output /tmp/anymal-train
python source/isaaclab_physicsnemo/examples/anymal_d/deploy.py --artifact /tmp/anymal-train/artifact
python source/isaaclab_physicsnemo/examples/anymal_d/train_policy.py --output /tmp/ppo-anymal-learned \
  --artifact /tmp/anymal-train/artifact
```

The ANYmal-D example follows the FlashNeRD locomotion experiment and the PhysicsNeMo Robotics G1
application, using PhysicsNeMo Robotics library components only:

- **Contact:** four fixed foot-sphere probes (`FixedRigidContactEncoder` with
  `expand_rigid_contact_shapes`), expressed in the base heading frame and flattened by
  `FixedContactFeatureAdapter` with the G1 channels. FlashNeRD uses fixed probes for this experiment and
  variable contact tokens only for platform climbing. Probes query per-world surfaces, so the teacher
  gives every environment its own ground plane.
- **Model:** `ParallelNeRD`, the FlashNeRD architecture, with the G1 geometry (`PARALLEL_NERD_CONFIG`),
  conditioned on the applied joint efforts and the probe features; 5 % trajectory validation split.
- **Data:** `AnymalDExplorationEnvCfg` randomizes base poses, base velocities, and joint configurations;
  uniform joint-position targets within 0.75 rad of the default pose change every 20 ms; episodes last
  at most 2 s and keep falls.
- **Timing:** the learned frame is Isaac Lab's 5 ms physics step. The ANYdrive actuator network runs once
  per physics step, so a 20 ms physics step degrades the actuator itself: with it, even PPO in MuJoCo
  Warp fails to learn to walk. FlashNeRD instead learns the actuator response inside a 20 ms frame.

Both ANYmal-D scripts accept `--policy` with a TorchScript policy exported by the RSL-RL `play.py`
script, to add a walking policy to the exploration targets.

### What the scripts do

- `record.py` steps the exploration configuration with random actions and writes every episode,
  including asynchronous per-environment resets.
- `train.py` trains a NeRD for a fixed number of updates and packages a hash-verified artifact. It prints
  the held-out one-step error and, for effort-only Cartpole models, the recursive-rollout error `E_H` of
  the FlashNeRD evaluation (`evaluate_rollout_nrmse`) for `H` = 1, 15, 100, and 200.
- `deploy.py` runs a MuJoCo Warp rollout and a learned rollout from the same seed and actions,
  checks the learned environment with `preflight_learned_physics`, and prints the open-loop joint
  error of every environment that has not been reset.

Default settings complete in a few minutes per stage on one GPU and only demonstrate the workflow.

### A usable Cartpole model

With FlashNeRD-scale data and a longer fit, the Cartpole model is accurate enough to train a policy
that transfers to MuJoCo Warp:

```bash
python source/isaaclab_physicsnemo/examples/cartpole/record.py --output /tmp/cartpole-data \
  --num-envs 1024 --steps 1500
python source/isaaclab_physicsnemo/examples/cartpole/train.py \
  --dataset /tmp/cartpole-data --output /tmp/cartpole-train --max-updates 30000 --batch-size 1024
```

The dataset holds about 10,800 episodes and 1.5 million frames; recording takes about 11 minutes and
training about 70 minutes on one GPU. Reference results:

| Held-out rollout error | E1 | E15 | E100 | E200 |
| --- | --- | --- | --- | --- |
| This model, 30,000 updates | 1.5e-4 | 8.5e-4 | 1.3e-2 | 8.4e-2 |
| FlashNeRD paper, 5,000,000 updates | 4.4e-5 | 1.7e-4 | 3.2e-3 | 6.0e-2 |

The deployed model stays within 0.016 m or rad of MuJoCo Warp joint positions after 0.5 s of open-loop
random actions. A PPO policy trained only in the learned environment (`train_policy.py --artifact`)
balances the MuJoCo Warp Cartpole in every evaluation episode, with the same return (4.95) as a policy
trained in MuJoCo Warp, for two seeds.

The per-episode action-scale range matters. With a fixed scale of 0.5 (about 50 N) the model reached a
lower E200 of 0.023, but the PPO policy learned to exploit its errors at larger efforts and succeeded in
only 55 % of MuJoCo Warp episodes. Recorded data must cover the command range that controllers use.

### ANYmal-D

The FlashNeRD-scale run:

```bash
python source/isaaclab_physicsnemo/examples/anymal_d/record.py --output /tmp/anymal-data \
  --num-envs 1024 --steps 3000
python source/isaaclab_physicsnemo/examples/anymal_d/train.py \
  --dataset /tmp/anymal-data --output /tmp/anymal-train --max-updates 60000
for physics in "" "--artifact /tmp/anymal-train/artifact"; do
  python source/isaaclab_physicsnemo/examples/anymal_d/train_policy.py --output /tmp/ppo-anymal $physics
done
```

The dataset holds 8,192 episodes and 3.1 million 5 ms frames; recording takes about 8 minutes and
training the 3.5M-parameter model about 70 minutes. Reference results:

- One-step validation normalized delta RMSE 0.26. After 2 s of open-loop random targets, joint position
  error 0.11 rad, joint velocity error 1.7 rad/s, and base height error 0.025 m.
- A policy trained in MuJoCo Warp keeps most of its reward in the learned environment (0.0149 against
  0.0174 per step, linear-velocity tracking error 0.40 against 0.36 m/s), but falls in 17 % of its
  learned episodes against 2 % in MuJoCo Warp.
- PPO trained only in the learned environment (300 iterations, 4,096 environments) yields policies that
  walk in MuJoCo Warp but track commands worse than MuJoCo Warp training:

| Training physics, seed | MuJoCo Warp episode return | MuJoCo Warp fall rate |
| --- | --- | --- |
| MuJoCo Warp, 42 / 43 | 17.7 / 13.5 | 0.8 % / 3.9 % |
| Learned, 42 / 43 | 6.6 / 5.2 | 2.0 % / 31 % |

FlashNeRD reports parity (28.0 against 27.8 in its reward). The remaining differences from its setup are
the 5 ms frame with four times as many autoregressive steps and a 50 ms history window, a smaller model
(3.5M against 8.88M parameters), and the PhysicsNeMo Robotics library codec, which keeps the base yaw and
has no body-frame gravity input, where the G1 application adds an application-level heading codec.

## Library

| API | Purpose |
| --- | --- |
| `physics.PhysicsNeMoSolverCfg`, `physics.NewtonPhysicsNeMoManager` | Newton solver configuration and manager that run an artifact |
| `physics.validate_state_ownership` | Check that a codec owns every dynamic Newton state row |
| `physics.validate_effort_control` | Check that applied joint efforts carry every actuation command |
| `tasks.CartpoleTeacherEnvCfg`, `tasks.AnymalDFlatTeacherEnvCfg` | Teacher configurations whose physics an artifact can replace |
| `tasks.CartpoleExplorationEnvCfg`, `tasks.AnymalDExplorationEnvCfg` | Teachers with task-independent initial states for data collection |
| `tasks.anymal_d_foot_contacts`, `tasks.anymal_d_dynamics_features` | ANYmal-D fixed foot probes and model inputs built from PhysicsNeMo Robotics encoders |
| `TrajectoryDatasetRecorder` | Write the episodes of a running environment to a Zarr dataset |
| `SmoothRandomActions`, `UniformRandomActions` | Temporally correlated, or uniform held, exploration actions |
| `TrajectoryDatasetRecorder(recorded_features=...)`, `RecordedFeature` | Record additional transition inputs such as fixed contact probes |
| `train_dynamics_artifact`, `TrainingSettings`, `PARALLEL_NERD_CONFIG` | Train and package a NeRD or ParallelNeRD artifact |
| `apply_learned_physics` | Replace the teacher solver of a configuration with an artifact |
| `preflight_learned_physics` | Check a live environment against the deployment contract |
| `NewtonBinding`, `execution_spec_from_manager` | Simulation identity and live buffers of an environment |
| `record_joint_rollout`, `compare_rollouts` | Measure the open-loop error of a learned rollout |
| `evaluate_rollout_nrmse`, `HorizonError` | Prefix-averaged recursive-rollout error on recorded trajectories |
| `isaaclab_model_signature`, `isaaclab_canonical_view` | Solver- and environment-count-independent model identity |

Deploy a recorded task with the same configuration it was recorded with:

```python
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_physicsnemo import apply_learned_physics, preflight_learned_physics
from isaaclab_physicsnemo.tasks import ANYMAL_D_CONTINUITY, AnymalDFlatTeacherEnvCfg

cfg = AnymalDFlatTeacherEnvCfg()
artifact = apply_learned_physics(cfg, "/tmp/anymal-train/artifact", continuity=ANYMAL_D_CONTINUITY)
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
assert preflight_learned_physics(env.sim.physics_manager, artifact).passed
```

`apply_learned_physics` only fills the solver configuration. The equivalent explicit configuration is:

```python
from isaaclab_newton.physics import NewtonCfg

from isaaclab_physicsnemo.physics import PhysicsNeMoSolverCfg

cfg.sim.dt = artifact_frame_dt
cfg.sim.physics = NewtonCfg(
    solver_cfg=PhysicsNeMoSolverCfg(artifact_path="/tmp/anymal-train/artifact", continuity="off"),
    num_substeps=1,
    use_cuda_graph=False,
    default_shape_cfg=...,  # the teacher's shape settings, which are part of the model identity
)
```

## Adapting a task

A teacher configuration must satisfy the learned solver contract:

- One environment step is one physics frame: `decimation = 1`. Choose `sim.dt` as the control
  period and let the teacher solver use substeps.
- The scene has no contact-force sensor, IMU acceleration, or joint-wrench output, and no reward or
  termination reads them.
- No event changes model properties (material, mass, inertia, center of mass, joint parameters) or
  applies external wrenches. State resets and velocity writes are supported.
- Joint efforts are the only actuation: explicit actuators or effort control, no implicit position
  drives (`joint_target_ke` must be zero). Implicit joint damping with zero velocity targets is
  allowed because it depends only on the state.
- Every environment replicates the same robot, and the teacher solver uses its internal contacts
  or no contacts, so no Newton collision pipeline is part of the recorded execution.
- Articulations are trees that Newton forward kinematics rebuilds: one root, enabled joints, no
  loop-closing or cable joints, and identity anchors on free roots.
- Floating-base robots use `remove_root_translation="planar"` when recording and training, so the
  per-environment origin does not reach the model.

Recording and deployment may use different environment counts.

## Behavior the integration provides

These behaviors make current Isaac Lab scenes deployable with PhysicsNeMo Robotics artifacts. The
last two rows work around PhysicsNeMo Robotics behavior and can be removed when it changes upstream.

| Isaac Lab or PhysicsNeMo Robotics behavior | Handling |
| --- | --- |
| Isaac Lab labels replicated entities with `/World/envs/env_<i>/...` prim paths, while PhysicsNeMo Robotics removes only a leading `env_<i>/` | `isaaclab_model_signature` fingerprints a canonical view with the prefix rewritten |
| The MuJoCo Warp teacher registers `mujoco:*` custom attributes that the learned manager does not | The signature ignores solver-private attribute namespaces |
| USD-derived local transforms differ in their last bits with the environment grid | The signature rounds transform attributes to `SIGNATURE_SIGNIFICANT_DIGITS` (6) significant digits and flushes values below `SIGNATURE_ZERO_TOLERANCE` (1e-8) times the largest magnitude of the same attribute to zero. Other floating attributes stay exact, because PhysicsNeMo Robotics derives semantics such as continuous joints from exact thresholds |
| A ground plane replicated into every environment sits at a different origin in every world | The signature places every static plane at its point closest to the world origin |
| PhysicsNeMo Robotics fixed-probe geometry fingerprints use the raw model arrays | Probes are built, and artifacts rebuilt by the manager, inside `isaaclab_canonical_view` |
| Isaac Lab notifies the solver with `JOINT_PROPERTIES` after every fixed-root pose write ([#7777](https://github.com/isaac-sim/IsaacLab/pull/7777)) | The manager drops the notification while the joint anchors `joint_X_p` and `joint_X_c` are unchanged and rejects every other model change |
| Isaac Lab applies external body wrenches through `State.body_f`, which the learned model has no input for | The manager fails a step that carries an external body wrench instead of silently ignoring it |
| Implicit position drives act on targets that never appear in `control.joint_f` | Recording and deployment reject nonzero `joint_target_ke` |
| A deployed solver starts each episode with a repeated first frame as history | Training includes these cold-start windows |
| The PhysicsNeMo Robotics joint codec normalizes a free-root quaternion again when writing a prediction, so `continuity="verify"` rejects the solver's own output by one floating-point unit | Floating-base tasks deploy with `continuity="off"` (`tasks.ANYMAL_D_CONTINUITY`); Isaac Lab still reports every reset through the solver reset mask |
| PhysicsNeMo Robotics compares model fingerprints bit for bit | The manager passes the rounded Isaac Lab signature to `DynamicsArtifact.as_solver` |

## Tests

```bash
.venv/bin/python -m pytest source/isaaclab_physicsnemo/test
```

Tests skip when PhysicsNeMo Robotics is not installed. The CPU tests cover model identity, state
ownership, effort control, rollout comparison, and exploration. The CUDA Cartpole tests check that every
stage keeps the data semantics, independently of model accuracy:

| Test | Contract |
| --- | --- |
| `test_learned_cartpole_round_trip` | Record, train, and deploy through `NewtonCfg` with a different environment count; preflight passes; a mismatched physics step is rejected |
| `test_recorded_transitions_replay_on_the_teacher` | A recorded state and effort stepped by MuJoCo Warp reproduce the recorded next state, which pins the recording time alignment |
| `test_deployed_solver_matches_the_offline_operator` | Every deployed learned step equals the offline artifact operator on the same states, efforts, and resets, which pins state, effort, history, and reset handling |
| `test_memorized_trajectories_track_the_teacher` | A model fitted to a few episodes tracks them after deployment, which pins how training pairs states with efforts |

The CUDA tests take about two minutes, most of it in the memorization fit.
