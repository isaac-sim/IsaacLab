:orphan:

.. _microduck-pollen-tasks:

MicroDuck contributed skills
============================

MicroDuck skill recipes in ``isaaclab_tasks.contrib.microduck`` use Newton MJWarp, hosted
robot assets, native BAM servos and RSL-RL. The ``-Pollen`` walking IDs select the source-matched
recipes; :ref:`the other walking tasks <microduck-walking-training>` use the AGILE-inspired recipe.

.. list-table:: Skill tasks (18 registrations, 33 configurations)
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

These 33 configurations supplement the two AGILE-inspired walking registrations and their backlash
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
plus each corresponding ``_backlash.usd`` file. Isaac Lab downloads and caches the USDs
during normal asset loading.

Task-specific robot configurations define the electrical torque ceiling and viscous damping.
Full-contact variants use ``max_current=0`` and an approximately 1.06755316 Nm joint ceiling.
Walking, curated-contact and roller variants use a 1.75 A current limit. See
``source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/UPSTREAM.md`` for source attribution
and model licensing.

VelStand training and validation
--------------------------------

:doc:`microduck_velstand` describes the pinned walking warm start, two frozen teachers,
post-PPO distillation pass, checkpoint lifecycle and standard playback/export workflow. The
VelStand runner resolves teacher weights into a private cache automatically.

Evaluate recovery policies on held-out initial poses. Report foot-supported recovery,
retention, latency and falls separately for each pose family.
