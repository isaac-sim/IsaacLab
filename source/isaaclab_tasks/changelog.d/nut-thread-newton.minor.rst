Added
^^^^^

* Added Newton-backend support to :class:`~isaaclab_tasks.direct.factory.FactoryEnv`.
  Run ``Isaac-Factory-*-Direct-v0`` tasks under the Newton physics backend with
  ``presets=newton``; the existing PhysX path is unchanged. New helper modules:

  * :mod:`isaaclab_tasks.direct.factory.factory_control_newton` adapts
    :class:`newton.selection.ArticulationView` to the J/M shapes the
    Factory OSC math (``factory_control.compute_dof_torque``) expects, and
    folds ``joint_armature`` into the mass-matrix diagonal.
  * :mod:`isaaclab_tasks.direct.factory.factory_newton_setup` houses
    procedural Newton-only post-load asset patches (POSITION-mode override,
    parser-body gravity compensation, finger hydroelastic SDFs, contact-stack
    tuning).
