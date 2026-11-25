:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-neural-robot-dynamics:

Training on neural robot dynamics
=================================

The optional ``newton-neural`` integration records robot dynamics, fits a model and runs policies
through Newton's existing lifecycle. Install it with ``uv sync --extra newton-neural``.

Configuration ownership
-----------------------

The neural package follows Isaac Lab's functional boundaries: model definitions at the root,
solver integration in ``physics/``, training configuration in ``rl/`` and recorder terms/configuration
in ``record/``. The scripts own task resolution and compose those pieces before simulation starts;
the neural library does not import task registrations or resolve presets.

* ``NeRDCfg`` defines the model architecture and physical inputs. Task-specific model definitions
  such as ``AnymalCNeRDCfg`` are independent of recording, training and inference.
* ``NeRDTrainingCfg`` owns optimizer, batch, noise and validation settings.
* Recording uses an ordinary task environment configuration containing exploration actions,
  reset/termination events and ``RecorderManager`` terms. The standard environment owns stepping.
* ``NeRDSolverCfg`` and ``MJWarpNeRDSolverCfg`` own checkpoint selection, articulation selection
  and inference settings. The checkpoint persists the model definition and its physical contract.

Task-local model and fitting configurations live in ``dynamics/``. Recording environment
configurations and exploration terms live in ``record/``; policy agents stay in ``agents/``.

Tasks register ``nerd_cfg_entry_point``, ``nerd_training_cfg_entry_point`` and
``nerd_recording_cfg_entry_point`` independently. The normal task resolver composes the selected
recording environment with the model definition; it does not reconstruct another environment
just to discover output paths.

Record and train
----------------

Paths are explicit. The recording produces ``model.pt`` (the untrained model), native trajectory
shards, ``meta.json`` and resolved settings under ``params/``.

.. code-block:: bash

   uv run --extra newton-neural python scripts/neural/record.py \
       --task Isaac-Reach-Franka --dataset logs/neural/reach_franka/dataset \
       --num_envs 4096 --num_frames 2048

   uv run --extra newton-neural python scripts/neural/train.py \
       --task Isaac-Reach-Franka --dataset logs/neural/reach_franka/dataset \
       --checkpoint logs/neural/reach_franka/flashnerd.pt

``agent.backbone_spans=[1,1]`` configures the model when recording. During training, the dataset's
saved model is authoritative; ``agent.learning_rate=0.001`` or ``agent.batch_size=1024`` configures
optimization independently. An output checkpoint is saved before the validation report.

Franka explores the arm in free space using mixed joint targets. ANYmal explores moving and tilted
poses using held joint targets and ground probes. Allegro records independent single-solver-step
samples with free-acceleration labels. Its recording configuration declares the reduced timestep
and constraint capacity explicitly.

Recording architecture
----------------------

Recording uses the existing ``RecorderManager`` and the normal environment step. A dynamics recorder
term copies state and contact inputs before physics, then returns a complete frame with consumed controls,
free-motion labels and validity after physics. The manager stores each world's episode and exports it
before automatic reset. Exploration actions own command generators; reset events write sampled states,
and termination terms declare the exploration duration and reject divergent state.

The native output handler implements the recorder's dataset interface and batches exported episodes into
``TrajectoryWriter`` shards. Each episode is an independent native world; shorter episodes are padded with
invalid frames. Training and validation split by those independent episodes. There is no intermediate
HDF5 conversion, alternate simulation loop or dependency on PR #9121.

Run policies
------------

Train policies through the existing RL entrypoint and the task's declared physics preset:

.. code-block:: bash

   uv run --extra newton-neural isaaclab train --rl_library rsl_rl \
       --task Isaac-Reach-Franka physics=newton_nerd

Franka reach and ANYmal C flat use ``physics=newton_nerd``. Allegro uses
``physics=newton_mjwarp_nerd``. Both use the ordinary Newton manager and backend.
For standalone dynamics, the task preset declares one recorded control period as ``sim.dt`` and
sets ``decimation=1`` and ``num_substeps=1``. The environment control period is preserved, and
simulation time advances by the declared timestep. The free-motion model advances each MuJoCo Warp
solver substep, retaining numerical contacts and object dynamics. Native stepping rejects a timestep
that differs from the checkpoint frame.

Each solver adapter loads its checkpoint once during normal solver construction. It does not consult
the global simulation singleton or introduce a manager subclass or a cached-model lookup.
NewtonBackend owns initialization, contacts, graph capture, resets and pending model changes.
The native solver consumes the controls declared by its model directly from Newton's control buffers.

The physical-model contract checks robot masses, inertias, centres of mass and armature. Free-motion
models also check drive gains. Binding separately checks joint structure, frames and gravity. A mismatch
requires recording and training a compatible checkpoint. The nominal task configurations omit known
unsupported robot randomization. Runtime mutations retain the native solver's explicit warnings.
For bounded exploration, sampling uses the action term's declared bounds. Deployment task presets
declare matching clipping; the physics integration does not inspect environment action classes.

Standalone learned dynamics do not compute joint reaction wrenches or accelerations, and reject sensors
that require those values. Contact-force observables are penetration-based indicators, not calibrated
force measurements. External body forces do not affect the standalone model. Use free-motion coupling
when numerical contact with other moving bodies is required. Neural solvers currently reject coupled-solver
entries.

Evaluate an explicit policy
---------------------------

Policy evaluation uses the shared RSL-RL playback workflow. ``physics=`` selects evaluation
physics; ``--checkpoint`` selects the policy using the ordinary playback checkpoint resolver.

.. code-block:: bash

   uv run --extra newton-neural isaaclab play --rl_library rsl_rl --evaluate \
       --task Isaac-Reach-Franka --checkpoint logs/rsl_rl/reach_franka_nerd/RUN/model_100.pt \
       physics=newton_mjwarp

Evaluate additional checkpoints by invoking the same command again. Configuration resolves once,
as in ordinary playback. Supply matching ``agent.*`` overrides when the policy was trained with
settings that differ from the selected task defaults, including network dimensions and action clipping.
For example, ANYmal C policies trained with neural physics use ``agent.clip_actions=1.5``.
Saved agent YAML is not restored automatically.

The report counts each world's first episode once and reports reward, duration and termination
fractions. It does not reconstruct per-world command metrics from already-aggregated reset logs.

Migration
---------

The combined recipe API has been removed from this unmerged integration. Model configuration,
optimizer configuration and recording environments have separate registry entries as described above.
The old recording simulation wrappers and recorder-owned loop are removed. Dataset and output checkpoint
paths must be supplied explicitly. Existing checkpoints without Isaac Lab physical-contract metadata
must be recorded and trained again.
