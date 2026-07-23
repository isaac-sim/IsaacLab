Changed
^^^^^^^

* **Breaking:** Replaced the manual MJWarp and VBD presets in
  ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Cloth-Franka`` with proxy
  coupling. Import :mod:`isaaclab_contrib.custom_coupling` for the opt-in
  manual example. Both task configurations default to one environment until
  proxy resets preserve per-world coupling state.
