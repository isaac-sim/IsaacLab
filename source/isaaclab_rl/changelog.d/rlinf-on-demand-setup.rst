Added
^^^^^

* Added ``scripts/reinforcement_learning/rlinf/setup_rlinf.py``, an idempotent on-demand installer for
  the RLinf VLA post-training demo. It installs the pins that must bypass the dependency resolver,
  builds ``pytorch3d``, clones Isaac-GR00T at a pinned commit, and downloads the pretrained
  checkpoint, so none of them have to be baked into an Isaac Lab environment or container image.

Fixed
^^^^^

* Fixed ``--checkpoint`` being silently ignored by the RLinf play entrypoint. It was written to
  ``runner.eval_policy_path``, which RLinf does not read; the RL-finetuned weights are now loaded
  through the RLinf extension instead. A ``global_step_<N>`` directory is resolved to the
  ``full_weights.pt`` it contains.
* Fixed relative ``model_path`` values in RLinf configs resolving against the Ray worker's working
  directory instead of the launcher's. Both the train and play entrypoints now make the path absolute
  before the config reaches the workers.
