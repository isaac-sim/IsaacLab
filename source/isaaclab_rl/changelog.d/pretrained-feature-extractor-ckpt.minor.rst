Added
^^^^^

* Added publication and retrieval of the companion checkpoints a task trains beside its policy, so
  ``--checkpoint pretrained`` also provides them. A task declares them in its :func:`gym.register`
  kwargs as ``"companion_checkpoints": {<name>: <run glob>}``; each is published beside the policy
  as ``<policy stem>_<name><extension>`` and downloaded into the same directory, which playback
  uses as the run log directory. :func:`~isaaclab_rl.utils.pretrained_checkpoint.get_companion_checkpoints`
  and :func:`~isaaclab_rl.utils.pretrained_checkpoint.get_companion_checkpoint_path` expose the same
  declaration to other tooling.

Changed
^^^^^^^

* Changed the pre-trained checkpoint cache to give every published checkpoint its own directory under
  ``.pretrained_checkpoints/<rl_library>/``. Playback treats that directory as the run log directory, so
  recorded videos and exported policies no longer overwrite each other across tasks, and a task's
  companion checkpoints sit beside the policy they belong to.

* Changed ``train_and_publish_checkpoints.py --publish_checkpoint`` to fail a job whose declared
  companion was not collected, instead of publishing the policy alone. A task that declares a
  companion needs it to play, so such a bundle failed on load after being reported as published.
