Added
^^^^^

* Added retrieval and publication of the feature-extractor checkpoint stored beside a published policy
  checkpoint, so ``--checkpoint pretrained`` also provides the trained CNN that vision tasks need.

Changed
^^^^^^^

* Changed the pre-trained checkpoint cache to give every published checkpoint its own directory under
  ``.pretrained_checkpoints/<rl_library>/``. Playback treats that directory as the run log directory, so
  recorded videos and exported policies no longer overwrite each other across tasks. Each declared
  checkpoint that is fetched records its own local copy, so a component finds it whatever log directory
  its workflow derives.

* Changed ``train_and_publish_checkpoints.py --publish_checkpoint`` to fail a job whose declared
  checkpoint was not collected, instead of publishing the policy alone. A component that declares a
  checkpoint needs it to play, so such a bundle failed on load after being reported as published.
