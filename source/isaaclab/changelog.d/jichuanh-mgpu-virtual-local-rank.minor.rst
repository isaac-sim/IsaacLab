Added
^^^^^

* Added automatic single-GPU-per-worker execution to the multi-GPU launchers when the selected presets
  include OVPhysX, which otherwise hangs runs on more than one GPU whenever an RTX renderer shares the
  process. Every worker sees its own GPU as ``cuda:0`` and reports ``LOCAL_RANK=0`` while this is active,
  so use the global rank to name per-rank files. This is applied automatically and needs no flag; it will
  be removed once the pinned OVPhysX version contains the fix.
