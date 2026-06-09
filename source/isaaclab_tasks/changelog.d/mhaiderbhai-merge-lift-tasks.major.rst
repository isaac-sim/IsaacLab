Changed
^^^^^^^

* **Breaking:** Consolidated the rigid and soft Franka lifting tasks into a single
  :mod:`isaaclab_tasks.core.lift` package. The former ``isaaclab_tasks.core.lift_franka_soft``
  package moved under :mod:`isaaclab_tasks.core.lift.config.franka_soft`, alongside the existing
  rigid ``franka`` config, to keep the rigid and soft variants separated. Update imports such as
  ``from isaaclab_tasks.core.lift_franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg`` to
  ``from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg``.
  The deformable MDP terms were merged into the shared :mod:`isaaclab_tasks.core.lift.mdp` package
  (rather than a separate per-variant ``mdp``); import them from there, e.g.
  ``from isaaclab_tasks.core.lift.mdp import deformable_lifted``.
* **Breaking:** Renamed the lift Gym environment IDs to drop the ``-v0`` version suffix. Update
  ``gym.make`` / ``--task`` calls:

  * ``Isaac-Lift-Cube-Franka-v0`` → ``Isaac-Lift-Cube-Franka``.
  * ``Isaac-Lift-Cube-Franka-Play-v0`` → ``Isaac-Lift-Cube-Franka-Play``.
  * ``Isaac-Lift-Soft-Franka-v0`` → ``Isaac-Lift-Soft-Franka``.
  * ``Isaac-Lift-Cloth-Franka-v0`` → ``Isaac-Lift-Cloth-Franka``.

Fixed
^^^^^

* Renamed the ``Isaac-Lift-Cloth-Franka`` physics preset from the misspelled ``newton_mjwarp_vdb``
  to ``newton_mjwarp_vbd``, matching the soft-body task and the underlying VBD solver. The cloth
  ``RewardsCfg``, which duplicated the soft task's rewards verbatim, is now inherited instead of
  redefined.
