Added
^^^^^

* Added ``instance_segmentation_fast`` and ``instance_id_segmentation_fast`` rendering test
  presets to the rendering test utilities for Cartpole, ShadowHand, and Dexsuite environments,
  covering both the OVRTX and Isaac RTX renderers. Because instance IDs are non-stable across
  runs, golden images are saved with full RGBA colors but comparison is restricted to the alpha
  channel only, which encodes the instance mask shape reliably. SSIM is also disabled for these
  data types; only the per-pixel L2 gate is used to decide pass/fail.
