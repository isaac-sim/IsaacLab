Changed
^^^^^^^

* Capped the effort limit and set a reflected inertia on the GR1T2 pick-place passive hand
  actuator groups under the ``newton_mjwarp`` preset, matching the Newton-validated
  ``panda_finger2_passive`` group on the Franka lift task. Zero stiffness and damping disable the
  second PD drive on a mimic follower, but leaving the effort limit uncapped is the failure the
  Shadow Hand asset documents as "an uncapped effort limit on either end diverges to NaN".
