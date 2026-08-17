Added
^^^^^

* Added ``--virtual_local_rank`` to the multi-GPU launchers, which gives every worker a single GPU as
  ``cuda:0``. It works around OVPhysX up to 0.5.10 selecting the wrong CUDA device when OVRTX shares the
  process, which hangs ``presets=ovphysx,ovrtx`` runs on more than one GPU. Every worker reports
  ``LOCAL_RANK=0`` while it is set, so use the global rank to name per-rank files.
