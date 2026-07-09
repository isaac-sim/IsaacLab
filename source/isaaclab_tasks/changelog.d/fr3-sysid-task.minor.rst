Added
^^^^^

* Added the ``contrib/sysid`` task family with ``Isaac-Sysid-Franka-FR3-v0``: a fixed-base
  Franka FR3 replay environment (Newton/mjwarp default preset, zero-g compensated-plant
  model) used by ``scripts/sysid/fit.py`` to fit per-joint implicit-actuator
  ``{stiffness, damping}`` against real chirp datasets via CMA-ES, with
  provenance-driven command-shaping reconstruction (clamp + EMA + Ruckig) of the
  ``franka_fr3`` ros2_control driver.
