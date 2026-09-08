Added
^^^^^

* Added ``scripts/reinforcement_learning/rlinf/setup_rlinf.py``, an idempotent on-demand installer for
  the RLinf VLA post-training demo. It installs the pins that must bypass the dependency resolver,
  builds ``pytorch3d``, clones Isaac-GR00T at a pinned commit, and downloads the pretrained
  checkpoint, so none of them have to be baked into an Isaac Lab environment or container image.

Fixed
^^^^^

* Fixed ``--rl_model_path`` being silently ignored by the RLinf play entrypoint. The flag was parsed
  and read by the RLinf extension, but never written into the config, so evaluating an RL-finetuned
  checkpoint silently evaluated the base model instead.
* Fixed relative ``model_path`` values in RLinf configs resolving against the Ray worker's working
  directory instead of the launcher's. Both the train and play entrypoints now make the path absolute
  before the config reaches the workers.
