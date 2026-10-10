Changelog
---------

.. towncrier release notes start

1.0.1 (2026-10-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the Warp twin of :func:`~isaaclab_tasks.core.velocity.mdp.feet_slide` reading the articulation body velocities
  with the contact sensor's body indices and ignoring the body selection of ``asset_cfg``. It now indexes each with its
  own selection, so the reward is correct when the sensor and the articulation index the selected bodies
  differently, for example under ``body_ordering`` or with a sensor that tracks only the feet.


1.0.0 (2026-10-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Warp MDP twins for :func:`~isaaclab_tasks.core.locomotion.mdp.terminated_penalty` and
  :class:`~isaaclab_tasks.core.locomotion.mdp.survival_success_rate`, so ``Isaac-Ant`` and
  ``Isaac-Humanoid`` run under ``--frontend warp``.

Changed
^^^^^^^

* Changed the warp locomotion environment to resolve ``joint_gears`` by joint name expression,
  matching the stable ``LocomotionDirectEnv``. This drops the Newton-only restriction that the
  previous backend-keyed lookup imposed. Joints the table does not match keep a unit gear, as they do
  in the manager-based action and reward terms.
* Changed the warp locomotion environment to implement the same MDP as the stable direct and
  manager-based ant and humanoid tasks, so ``--frontend warp`` trains against the same problem. It
  now observes the feet joint wrenches, randomizes the joint state on reset, scales the continuous
  reward terms by the environment step interval, weighs the energy and joint-limit penalties by the
  per-joint gear ratio, and applies the death cost as a one-off terminal penalty.
* **Breaking:** Renamed the multi-word robot names in the experimental Warp locomotion environment
  IDs to CamelCase (``-`` is reserved for task-name aspects), keeping the ``-Warp-v0`` suffix. For
  example ``Isaac-Velocity-Flat-Anymal-C-Warp-v0`` → ``Isaac-Velocity-Flat-AnymalC-Warp-v0`` and
  ``Isaac-Velocity-Flat-Unitree-Go2-Warp-v0`` → ``Isaac-Velocity-Flat-UnitreeGo2-Warp-v0``
  (similarly ``AnymalB``, ``AnymalD``, ``UnitreeA1``, ``UnitreeGo1``).
* Changed experimental Newton task presets to rely on iterative MuJoCo Warp
  line search.
* **Breaking:** Removed the ``*_PLAY`` environment configuration classes and the ``-Play`` gym task
  registrations. Play-mode overrides are now defined by overriding ``play_mode`` on the training
  environment configuration and are applied automatically by the play scripts. Use the training task
  id with ``play.py``, and pass ``--train_env_cfg`` to play the training configuration as-is.
* **Breaking:** Changed the package layout to mirror :mod:`isaaclab_tasks.core`
  (``isaaclab_tasks_experimental.core.<task>``), replacing the previous
  ``manager_based``/``direct`` split. Update imports of the old paths to the
  ``core`` equivalents (e.g. ``isaaclab_tasks_experimental.core.cartpole.mdp``).
* Changed the direct Warp env modules and classes to the ``<task>_warp_env`` /
  ``<Name>WarpEnv`` convention so ``--frontend warp`` resolves them by name:
  renamed ``ant_env_warp`` to ``ant_warp_env`` and ``InHandManipulationWarpEnv``
  (module ``inhand_manipulation_warp_env``) to ``ReorientDirectWarpEnv``
  (module ``reorient_warp_env``).

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``,
  ensuring standalone pip installs resolve local checkouts without a package index.
* **Breaking:** Removed all manager-based ``*-Warp-v0`` task registrations —
  ``Isaac-Cartpole-Warp-v0``, ``Isaac-Humanoid-Warp-v0``, ``Isaac-Ant-Warp-v0``,
  ``Isaac-Reach-Franka-Warp-v0``, ``Isaac-Reach-Franka-Warp-Play-v0``, and the
  velocity variants (``Isaac-Velocity-Flat-<Robot>-Warp-v0`` and their
  ``-Warp-Play-v0`` forms) — together with their environment configurations.
  Run the stable task ids with ``--frontend warp`` and
  ``presets=newton_mjwarp`` instead, e.g.
  ``--task Isaac-Velocity-Flat-AnymalD --frontend warp presets=newton_mjwarp``.
* **Breaking:** Removed the duplicated direct warp task registrations
  ``Isaac-Cartpole-Direct-Warp-v0``, ``Isaac-Ant-Direct-Warp-v0``,
  ``Isaac-Humanoid-Direct-Warp-v0``, and
  ``Isaac-Reorient-Cube-Allegro-Direct-Warp-v0`` together with their
  duplicated environment configurations; the frontend resolves their Warp
  environment classes by the mirrored naming convention, with
  ``warp_entry_point`` available as an optional override. Run the stable task
  ids with ``--frontend warp`` and ``presets=newton_mjwarp`` instead, e.g.
  ``--task Isaac-Cartpole-Direct --frontend warp presets=newton_mjwarp``.
* Removed the unregistered rough velocity warp configurations; rough-terrain
  warp tasks remain unsupported until :class:`~isaaclab.terrains.TerrainImporter`
  gains Warp APIs.

Fixed
^^^^^

* Fixed Warp reorientation goal-marker instances appearing across environment
  scene partitions.
* Fixed the Warp orientation-error twins reporting a smaller rotation error than the stable
  terms for non-unit quaternions, by computing the angle as ``2*atan2(|xyz|, |w|)`` instead of
  ``2*acos(|w|)``.
* Fixed the warp locomotion environment not logging ``Metrics/success_rate``, which both stable
  workflows report. The rate is now reduced on device and exposed as a tensor view, so the
  computation stays CUDA-graph capturable.

* Fixed the warp locomotion environment terminating one step earlier than the stable tasks and
  treating a torso below the negative termination height as a fall.
* Fixed the Warp reorientation reset sampling hand joint positions around the
  lower joint limit instead of between the lower and upper limits, matching the
  torch reorientation tasks.
* Used resolved callable defaults when constructing manager terms instead of duplicating defaults in constructors.
* Added the Warp implementation of ``feet_air_time_variance``, restoring Warp frontend support
  for Cassie and H1 tasks with the gait-symmetry reward enabled.
* Fixed the direct Warp Cartpole task to match the stable task's observations,
  reset ranges, termination condition, reward scaling, and scene configuration.
* Fixed the Warp Cartpole ``survival_success_rate`` twin to report the
  ``Metrics/success_rate`` value on-device instead of silently dropping the
  metric.
* Fixed ``Isaac-Humanoid-Direct`` under ``--frontend warp`` failing at env
  construction: the Warp locomotion env now resolves the per-physics-backend
  ``joint_gears`` dict (selecting the Newton ordering) like the stable
  ``LocomotionDirectEnv``, instead of passing the dict straight to ``wp.array``.
