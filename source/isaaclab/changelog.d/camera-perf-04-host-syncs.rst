Changed
^^^^^^^

* Removed per-step host synchronizations from the environment step: reset paths of the episode
  length, joint actions, event, reward, and termination managers, sensor reset masks, and circular
  buffers now fill on the device, and visualization-marker index validation runs only when a
  backend consumes the markers, once per index tensor.
* Skipped the camera mask device-to-host copy when ``update_period`` is zero, since every step
  marks all cameras outdated.
* Changed uniform ``add`` noise with a zero-width range to return its input without drawing samples.
