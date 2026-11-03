:orphan:

.. _microduck-pollen-tasks:

MicroDuck contributed skills
============================

MicroDuck's Pollen-derived skill recipes are included in ``isaaclab_tasks.contrib.microduck``.
No external task extension or separate project installation is required. They use Newton MJWarp,
prebuilt hosted robot assets, native BAM servos and RSL-RL. The existing
:ref:`walking tasks <microduck-walking-training>` retain their registrations and behavior.
The separately preserved imported walking recipes have the ``-Pollen`` qualifier.

.. list-table:: Imported tasks (18 registrations, 33 configurations)
   :header-rows: 1
   :widths: 60 20 20

   * - Task ID (all begin ``IsaacContrib-``)
     - Terrain
     - ``presets=backlash``
   * - ``Velocity-Flat-MicroDuck-Pollen``
     - Flat
     - Yes
   * - ``Velocity-Rough-MicroDuck-Pollen``
     - Rough
     - Yes
   * - ``Velocity-Flat-MicroDuck-Rollers``
     - Flat
     - Yes
   * - ``Velocity-Swizzle-MicroDuck``
     - Flat
     - Yes
   * - ``StandUp-Flat-MicroDuck`` / ``StandUp-Rough-MicroDuck``
     - Flat / rough
     - Yes
   * - ``SitStand-Flat-MicroDuck`` / ``SitStand-Rough-MicroDuck``
     - Flat / rough
     - Yes
   * - ``GroundPick-Flat-MicroDuck`` / ``GroundPick-Rough-MicroDuck``
     - Flat / rough
     - Yes
   * - ``VelStand-Flat-MicroDuck`` / ``VelStand-Rough-MicroDuck``
     - Flat / rough
     - Yes
   * - ``BallKick-Flat-MicroDuck``
     - Flat
     - Yes
   * - ``RollerCrouch-Flat-MicroDuck``
     - Flat
     - Yes
   * - ``RollerSlope-Flat-MicroDuck``
     - Slope
     - Yes
   * - ``Roulade-Flat-MicroDuck``
     - Flat
     - No
   * - ``RollerStandUp-Flat-MicroDuck``
     - Flat
     - No
   * - ``Spin-Flat-MicroDuck``
     - Flat
     - No

These 33 configurations supplement the two existing walking registrations and their backlash
presets. Use the normal CLI to inspect, train and play them:

.. code-block:: bash

   uv run isaaclab list_envs --keyword MicroDuck --show_presets
   uv run isaaclab train --task IsaacContrib-SitStand-Rough-MicroDuck \
       --rl_library rsl_rl presets=backlash
   uv run isaaclab play --task IsaacContrib-SitStand-Rough-MicroDuck \
       --rl_library rsl_rl --checkpoint /absolute/path/model_N.pt presets=backlash

Backlash is a domain preset on the same task ID. It selects fourteen serial play hinges and
output-side feedback; roller wheel hinges remain passive. Terrain sensing uses the source foot
sites and terrain-relative reset heights. SitStand retains the source seated recovery perturbations
and neck-strain hardening. Swizzle includes an optional normalization adapter that reproduces source rollout-statistics timing;
its default remains the native runner. Select all three class overrides together:

.. code-block:: bash

   uv run isaaclab train --task IsaacContrib-Velocity-Swizzle-MicroDuck --rl_library rsl_rl \
       agent.algorithm.class_name=isaaclab_tasks.contrib.microduck.velocity.agents.rollout_normalization:MicroDuckRolloutNormalizationPPO \
       agent.actor.class_name=isaaclab_tasks.contrib.microduck.velstand.agents.model:MicroDuckVelStandMLPModel \
       agent.critic.class_name=isaaclab_tasks.contrib.microduck.velstand.agents.model:MicroDuckVelStandMLPModel

Assets and physical settings
-----------------------------

All robot configurations load prebuilt files below
``ISAACLAB_NUCLEUS_DIR/Robots/PollenRobotics/MicroDuck``: ``microduck_walk.usd``,
``microduck_allcollisions.usd``, ``microduck_rollers.usd`` and ``microduck_fullcontact.usd``,
plus each corresponding ``_backlash.usd`` file. No source MJCF, local fallback or startup conversion
is required. The hosted asset sync is assumed available.

The Pollen recipe deployment clones retain their explicit electrical torque ceiling and viscous
damping; they do not change the original walking defaults. Full-contact variants retain a local
``max_current=0`` deviation and approximately 1.06755316 Nm joint ceiling. Pollen's pinned BAM
XL330 constructor uses 1.75 A, which the other variants retain. See
``source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/UPSTREAM.md`` for immutable source,
model licensing, embedded-drive and rotor-proxy provenance.

VelStand training and validation
--------------------------------

:doc:`microduck_velstand` describes the pinned walking warm start, two frozen teachers,
post-PPO distillation pass, checkpoint lifecycle and standard playback/export workflow. The
VelStand runner resolves teacher weights into a private cache automatically.

Training completion does not establish recovery quality. The imported final StandUp policies
have a known back-recovery failure under their final reset distribution, with and without backlash.
This integration preserves their formulation; it does not claim a repaired policy or a new
successful training. Evaluate recovery by initial orientation and report stable upright recovery,
height separately, latency and falls before accepting a candidate.
