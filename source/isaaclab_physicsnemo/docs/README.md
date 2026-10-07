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

Each task has three scripts under [`examples/`](../examples). Run them from the repository root.

### Cartpole

```bash
python source/isaaclab_physicsnemo/examples/cartpole/record.py --output /tmp/cartpole-data
python source/isaaclab_physicsnemo/examples/cartpole/train.py \
  --dataset /tmp/cartpole-data --output /tmp/cartpole-train
python source/isaaclab_physicsnemo/examples/cartpole/deploy.py --artifact /tmp/cartpole-train/artifact
```

### ANYmal-D on flat ground

```bash
python source/isaaclab_physicsnemo/examples/anymal_d/record.py --output /tmp/anymal-data
python source/isaaclab_physicsnemo/examples/anymal_d/train.py \
  --dataset /tmp/anymal-data --output /tmp/anymal-train
python source/isaaclab_physicsnemo/examples/anymal_d/deploy.py --artifact /tmp/anymal-train/artifact
```

Both ANYmal-D scripts accept `--policy` with a TorchScript policy exported by the RSL-RL `play.py`
script. Recording with a walking policy plus exploration noise covers locomotion states that random
joint targets alone rarely reach.

### What the scripts do

- `record.py` steps the teacher configuration with smooth random actions and writes every episode,
  including asynchronous per-environment resets.
- `train.py` trains a vector NeRD for a fixed number of updates and packages a hash-verified
  artifact. It prints the held-out one-step error.
- `deploy.py` runs a MuJoCo Warp rollout and a learned rollout from the same seed and actions,
  checks the learned environment with `preflight_learned_physics`, and prints the open-loop joint
  error of every environment that has not been reset.

Default settings complete in a few minutes per stage on one GPU. Use more environments, steps, and
updates for useful models.

Reference results with the default settings:

| Task | Dataset | Validation normalized one-step RMSE | Open-loop error after 0.5 s |
| --- | --- | --- | --- |
| Cartpole | 480 episodes, 64 envs | 0.028 | joint position 0.09, joint velocity 0.58 |
| ANYmal-D | 598 episodes, 128 envs | 0.36 | joint position 0.84, joint velocity 19, base height 0.07 m |

The ANYmal-D model conditions only on joint state and applied efforts. Ground contact is learned
implicitly from random exploration, which limits its accuracy. Recording with a walking policy, more
data and updates, and contact features such as PhysicsNeMo Robotics `FixedRigidContactEncoder` foot
probes are the next steps for a useful legged-robot model.

## Library

| API | Purpose |
| --- | --- |
| `physics.PhysicsNeMoSolverCfg`, `physics.NewtonPhysicsNeMoManager` | Newton solver configuration and manager that run an artifact |
| `physics.validate_state_ownership` | Check that a codec owns every dynamic Newton state row |
| `physics.validate_effort_control` | Check that applied joint efforts carry every actuation command |
| `tasks.CartpoleTeacherEnvCfg`, `tasks.AnymalDFlatTeacherEnvCfg` | Teacher configurations whose physics an artifact can replace |
| `TrajectoryDatasetRecorder` | Write the episodes of a running environment to a Zarr dataset |
| `SmoothRandomActions` | Temporally correlated exploration actions |
| `train_dynamics_artifact`, `TrainingSettings` | Train and package a control-conditioned vector NeRD |
| `apply_learned_physics` | Replace the teacher solver of a configuration with an artifact |
| `preflight_learned_physics` | Check a live environment against the deployment contract |
| `NewtonBinding`, `execution_spec_from_manager` | Simulation identity and live buffers of an environment |
| `record_joint_rollout`, `compare_rollouts` | Measure the open-loop error of a learned rollout |
| `isaaclab_model_signature` | Solver-independent model identity |

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
| USD-derived local transforms differ in their last bits with the environment grid | The signature rounds floating attributes to `SIGNATURE_SIGNIFICANT_DIGITS` (6) significant digits and flushes values below `SIGNATURE_ZERO_TOLERANCE` (1e-8) times the largest magnitude of the same attribute to zero |
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

Tests skip when PhysicsNeMo Robotics is not installed. The Cartpole round trip records, trains, and
deploys a small model on CUDA in about 15 seconds; the remaining tests run on CPU.
