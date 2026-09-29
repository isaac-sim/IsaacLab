Added
^^^^^

* Added runtime visual domain randomization with vectorized observation scheduling,
  episode-stable styles, partial-reset handling, action-chunk gating, cached reads,
  and explicit generation failure policies. Added passthrough and Cosmos backends,
  GPU worker processes, and explicit model residency lifecycle methods.

* Added public Cosmos3-Nano and four-step distilled YAML recipes. Supported public
  and GitLab guided-generation framework checkouts with either checkpoint, with or
  without mask guidance. Used the native mask API when available and an Isaac Lab
  sampler adapter otherwise. Public transfer requests ran sequentially when batched.

* Separated latent mask guidance from optional foreground pixel compositing. Added
  mask strength, all-step guidance by default, fractional area downsampling, and
  union or class-specific boundary erosion. Fixed semantic ID parsing and mask
  construction when compositing was disabled.

* Changed Cosmos defaults to the public checkpoint's 50-step, 720-tier recipe.
  Custom distilled exports required the matching fixed-step count and guidance
  settings; the supplied distilled recipe used four steps and camera-sized 640x480
  generation. Scoped camera-sized resolution overrides to each request. Added an
  explicit sampler shift and accepted the public Hugging Face checkpoint ID.

* Added an attention fallback for generation inside Kit, model residency cleanup,
  and a framework installer that retained Isaac Lab's torch environment. Documented
  setup, checkpoint-specific settings, guidance limitations, and the distinction
  between independent runtime images and offline reference-frame video generation.

Changed
^^^^^^^

* **Changed default behavior:** disabled foreground pixel compositing by default.
  Set ``CameraDRCfg.composite_foreground=True`` to retain source pixels exactly.
  Simplified the custom/distilled checkpoint setup instructions and documented
  zero boundary dilation and source projection through the final step as defaults.
