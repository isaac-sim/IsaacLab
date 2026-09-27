Changed
^^^^^^^

* Removed per-step host synchronizations from the environment step: reset paths of the episode
  length, joint actions, event, reward, and termination managers, sensor reset masks, and circular
  buffers now fill on the device. Marker IDs are validated by the backend that consumes them.
* Skipped the camera mask device-to-host copy when ``update_period`` is zero, since every step
  marks all cameras outdated.
* Changed uniform ``add`` noise with a zero-width range to return its input without drawing samples.
