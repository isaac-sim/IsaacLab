:orphan:

VelStand walking and recovery
=============================


VelStand follows the `merged upstream factory at bae39f5 <https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py>`__
and its `two-teacher post-PPO pass <https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/src/mjlab_microduck/tasks/distill.py>`__.
The proposed angular-velocity gate in upstream PR54 is excluded.

The task IDs are ``IsaacContrib-VelStand-Flat-MicroDuck`` and
``IsaacContrib-VelStand-Rough-MicroDuck``. Select ``presets=backlash`` for serial ±1°
gearbox play and firmware/output-encoder feedback. This is the upstream production
physical variant. Flat and non-backlash runs are comparisons. The actor remains 61
observations and 14 HOME-relative joint actions; the privileged critic has 76 inputs.

Checkpoints and training
------------------------


The selected reproducible lineage starts from the original walking checkpoint at 3750,
then trains 6000 **added** iterations with the original frozen walking and standing
teachers. Historical v8 had an additional multi-stage lineage; matching the merged
factory does not claim to reproduce those historical optimizer trajectories.

Public teacher artifacts are pinned to ``pollen-robotics/microduck-training`` revision
``64d28ce092253c509b51031ec7ccb53790a00d0d``, downloaded anonymously and verified by SHA256:

Walking warm start and anchor: ``velocity2/441tzs6d_3750/model_3750.pt``, SHA256 ``702c01bd02aeda1d7869390342e00298193e7071d475f3776e0a6eedfc5c8209``.

Frozen standing expert: ``microduck_stand/69u48n8l_9750/model_9750.pt``, SHA256 ``12bfcfddcdb4270e4cbbfb43554d0b14fbdad01263699cd97424e48e76b302fe``.


The training repository does not declare a weight license. Keep checkpoints in a private
cache; do not package or redistribute them. The upstream Apache notices and hosted-model attribution are retained in the native task package.
The VelStand runner automatically resolves both pinned teachers into this cache and verifies
their SHA256 hashes before use.

.. code-block:: bash

   export MICRODUCK_VELSTAND_CACHE=/absolute/private/cache/velstand
   uv run isaaclab train --task IsaacContrib-VelStand-Rough-MicroDuck \
       --num_envs 4096 --max_iterations 6000 --seed 42 --logger tensorboard presets=backlash


Omit ``presets=backlash`` for the non-backlash comparison; select the Flat ID for flat
terrain. The corresponding hosted prebuilt files are ``microduck_fullcontact_backlash.usd``
and ``microduck_fullcontact.usd``.

The algorithm automatically imports the pinned walk's actor, critic, normalization and
PPO Adam state, retaining its saved learning rate. Environment and runner counters start
at zero. Native checkpoints record terrain, selected presets, teacher hashes and the
environment counter. ``--checkpoint /absolute/path/model_N.pt`` resumes that same domain;
an incompatible terrain/preset is rejected. Loading resets the environment after restoring
its counter, applying curricula and refreshing observations before the next action. Invalid
physics samples contribute finite zero values per reward term before weighting totals and
episode metrics are accumulated, preserving other terms and prior episode history. The
task-local guard retains native reward signatures and reset hooks, including the paired
backlash selectors and head-error EMA. Bad samples cannot poison retained torque/velocity history. BC Adam starts fresh on resume, matching
upstream's checkpoint behavior. This implementation currently requires one GPU.

The native generic RSL-RL runner remains in use. Task-local models update normalization
on every next rollout observation, matching source RSL-RL 5.0, and suppress native 5.5's
extra end-of-update statistics pass. After each native PPO update, a dedicated actor-only
Adam pass teaches tilt>35° from the standing expert and tilt<25° from the walk anchor.
Standing body tilt/z commands route exactly as upstream. Only teacher inputs receive the
calibrated yaw remap and high-tilt joint masking; student commands remain unchanged.

Physical and task details
-------------------------


Only VelStand uses the new fullcontact asset family. It retains 70 source collision
geometries, of which 69 can produce contacts; the single mask-(2,2) power-support shell
has no valid partner. Housing, head and trunk sensors select 15/30/9 active meshes. The
head set intentionally retains upstream's historical right-hip ``bearing_roll`` inclusion.
Backlash serialization adds 14 collider-free 1 mg rotor proxies required by MJWarp; original
physical bodies retain their masses, centers of mass and full inertia tensors. This
14 mg addition is documented, rather than described as identical same-body MJCF mechanics.

This port's fullcontact servos disable the firmware current limiter, with measured native/public/
solver saturation 1.06755316 N·m. This is a local setting; Pollen's pinned BAM XL330 model retains
its 1.75 A default. Legacy current limits and assets are preserved.

Rough terrain uses source sole-site frames, three terrain rays per foot (center and
±4 cm), minimum clearance, source miss/backface rules and site velocity. The source
clearance target is 2 cm. Rough terrain contacts use solref(.04,1) and the source solimp.
Flat comparisons retain the existing ankle-frame sensing. Resets use terrain-relative
height and exclusive prone/crouch/sit/untouched buckets, including side, random-joint
and uniform-SO(3) starts. Protective costs and all recovery/body/topple schedules are
relative to the new run, not the walking checkpoint's old counter.

Evaluation and export
---------------------


Training completion alone does not establish walking/recovery quality. Validate held-out
walking/yaw/body commands, each spawn family and toppling pushes, reporting coverage,
recovery latency/re-falls, contact peaks/impulses, stalls and acceleration spikes. Keep
body-z tracking separate: it is an upstream limitation, not an acceptance claim.

Load a candidate with the standard playback command, which also exports the policy:

.. code-block:: bash

   uv run isaaclab play --task IsaacContrib-VelStand-Rough-MicroDuck --rl_library rsl_rl \
       --checkpoint /absolute/path/model_5999.pt presets=backlash


The runner resolves the pinned teachers from the private cache, downloading them if needed,
and loads the candidate before exporting. It writes ``exported/policy.pt`` and
``exported/policy.onnx`` beside the checkpoint before playback begins. The exports include
the actor's observation normalizer, with float32 input [1, 61] and output [1, 14]. Retain any
external-data sidecar the ONNX export produces. Compare outputs against the native actor on
recorded observations before transfer. Full-budget success and held-out evaluation remain
required before a PR is ready.
