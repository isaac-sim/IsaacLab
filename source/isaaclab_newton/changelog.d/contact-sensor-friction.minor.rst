Added
^^^^^

* Added aggregate and filtered friction force outputs to the Newton contact sensor.

Changed
^^^^^^^

* Changed Newton contact sensor normal-force outputs to exclude friction. Reconstruct the total
  contact force by adding ``net_normal_forces_w`` and ``net_friction_forces_w``.
