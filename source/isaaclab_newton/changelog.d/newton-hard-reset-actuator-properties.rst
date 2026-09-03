Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.Articulation` losing every actuator property on a hard
  reset. ``sim.reset(soft=False)`` re-finalizes the Newton model from the builder, and the
  rebuilt model carries the USD-authored joint drives, so the stiffness, damping, armature and
  effort limits written from :attr:`ArticulationCfg.actuators` were silently discarded. Measured
  on GR1T2: stiffness 4400 to 53026, damping 40 to 2148, armature 0.1 to 0.0 and the effort limit
  to ``inf``. Those values are outside what MJWarp can integrate, so the first commanded motion
  after the reset drove the whole articulation non-finite -- a zero command still looked healthy,
  which made the failure look like a task or tuning problem rather than a reset one. The
  ``PHYSICS_READY`` callback now re-applies the actuator configs in addition to rebinding the
  simulation data.

  Any Newton articulation whose authored USD drives differ from its actuator config was affected,
  and scripts following the ``record_demos.py`` reset sequence hit it on every reset.
