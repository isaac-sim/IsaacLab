Changed
^^^^^^^

* Acquired Newton viewer models from the simulation's shared backend registry instead of the
  physics manager. Viewers requested transforms and geometry directly through SDP; headless GL
  and RTX captures requested current arrays only when a frame was requested.
* Rebound GL, RTX, Rerun, and Viser resources after hard resets, including when picking was disabled.
