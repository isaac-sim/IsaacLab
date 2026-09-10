Fixed
^^^^^

* Fixed ``play --checkpoint pretrained`` on the Shadow Hand camera tasks, which raised
  ``ValueError: max() iterable argument is empty`` because the vision CNN the policy was trained with
  was never published. The feature extractor now declares the file it writes, so it is published
  beside the policy and fetched with it, and a missing checkpoint reports the directory that was searched.
  ``FeatureExtractorCfg.checkpoint_path`` takes a file handed over by the fetch, so playback resolves the
  CNN the same way under every RL workflow rather than searching a log directory each one derives differently.
