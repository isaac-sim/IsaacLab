Added
^^^^^

* Added Newton backend support for the multi-agent
  ``Isaac-Shadow-Hand-Over-Direct-v0`` (MAPPO/IPPO) env. Mirrors the
  single-agent Shadow Hand Newton port: per-hand
  :class:`ImplicitActuatorCfg`, ``shadow_hand_instanceable_newton.usd``,
  per-backend :class:`~isaaclab_tasks.utils.PresetCfg` wrappers for sim
  physics, scene cloning (``clone_in_fabric=False`` on Newton), the
  hand-over object (``RigidObjectCfg`` on both backends, dropping
  PhysX-only knobs on Newton), and the two robot configs. Selectable via
  ``--preset newton`` / Hydra preset resolution; PhysX behavior unchanged.

Fixed
^^^^^

* Fixed Newton training failing to learn the catch in
  ``Isaac-Shadow-Hand-Over-Direct-v0`` MAPPO. Two Newton-side
  :class:`~isaaclab.actuators.ImplicitActuatorCfg` overrides are added:

  * ``fingers`` (wrist + per-finger joints): ``stiffness=20.0`` /
    ``damping=2.0``, vs PhysX's ``5.0`` / ``0.5`` on wrists and
    ``1.0`` / ``0.1`` on fingers. PhysX layers
    ``fixed_tendons_props(limit_stiffness=30, damping=0.1)`` on top of
    the implicit drive and runs ``solver_position_iteration_count=8``
    per substep — both amplify the effective torque per unit nominal
    gain. Newton's MJWarp implicit-PD path has neither, so larger
    nominal gains are needed for comparable joint authority.
  * ``distal_passive`` (the four ``robot0_(FF|MF|RF|LF)J0`` joints):
    ``stiffness=10.0`` / ``damping=0.1``. The Newton USD bakes
    ``stiffness=286`` / ``damping=57`` on these joints from the
    MJCF→USD translation, which fights the ``MjcTendon`` coupling and
    bounces the ball. ``stiffness=10`` (~1/3 of PhysX
    ``limit_stiffness=30``) keeps the joints near-passive while the
    tendon constraint dominates. PhysX uses tendon coupling on these
    joints directly and does not need an analogous override.

  At iter 200 / 2048 envs, MAPPO ``Reward / Total reward (mean)``:
  PhysX baseline **246.7**, Newton at ``stiffness=1.0`` / ``damping=0.1``
  (no catch learned) **23.4**, Newton at the new gains **777.1**.
  Newton learns the catch reliably; longer runs and behavior-level
  comparison (catch / drop rate, ball trajectory) are follow-ups.
  PhysX path is unchanged.
