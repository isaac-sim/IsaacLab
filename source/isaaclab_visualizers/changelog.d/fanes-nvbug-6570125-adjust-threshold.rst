Changed
^^^^^^^

* Changed ``_assert_frames_remain_stable`` in the visualizer integration tests to accept a
  ``channel_diff_threshold``, and raised it for the two RTX pause comparisons only: 80 for the Kit
  viewport and 160 for the Kit tiled camera. RTX ResponsiveDenoising keeps refining a paused frame,
  so the residue is a few high-amplitude pixels that a per-pixel count cannot separate from real
  motion. The Newton ViewerGL pause checks rasterise without DLSS and keep the strict default of 50.
  This is a work-around for NVBUG 6570125 and should be reverted once the renderer fix ships.

Fixed
^^^^^

* Fixed the tiled camera pause assertion comparing the sensor's cached frame against itself, which
  made the check vacuous — it could not fail regardless of what the renderer did while paused. Both
  paused captures force a fresh render again, and the simulation app is pumped during the pause
  window.
