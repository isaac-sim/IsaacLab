Fixed
^^^^^

* Fixed the Digit velocity and loco-manipulation tasks declaring a ``newton_mjwarp``
  preset they cannot run. ``LocomotionVelocityRoughEnvCfg`` declares ``events.base_com``
  with a ``newton_mjwarp`` branch that disables the center-of-mass randomization, since
  Newton does not support it. Digit is PhysX-only, so that branch named no reachable
  backend and surfaced as a standalone ``presets=newton_mjwarp`` token that stripped the
  randomization from a PhysX run. The preset is now collapsed to its default on Digit.
  ``IsaacContrib-Velocity-Flat-Digit``, ``IsaacContrib-Velocity-Rough-Digit`` and
  ``IsaacContrib-Tracking-LocoManip-Digit`` no longer accept ``presets=newton_mjwarp``;
  it was never a backend switch on those tasks, and passing it only removed a
  randomization. Velocity tasks that do offer Newton are unchanged.
