Added
^^^^^

* Added retrieval and publication of the feature-extractor checkpoint stored beside a published policy
  checkpoint, so ``--checkpoint pretrained`` also provides the trained CNN that vision tasks need.

Changed
^^^^^^^

* Changed the pre-trained checkpoint cache to give every published checkpoint its own directory under
  ``.pretrained_checkpoints/<rl_library>/``. Playback treats that directory as the run log directory, so
  recorded videos and exported policies no longer overwrite each other across tasks.
