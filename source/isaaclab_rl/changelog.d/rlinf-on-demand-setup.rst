Added
^^^^^

* Added ``scripts/reinforcement_learning/rlinf/setup_rlinf.py``, an idempotent on-demand installer for
  the RLinf VLA post-training demo. It installs the pins that must bypass the dependency resolver,
  builds ``pytorch3d``, clones Isaac-GR00T at a pinned commit, and downloads the pretrained
  checkpoint, so none of them have to be baked into an Isaac Lab environment or container image.
* Added ``--gr00t {n15,n17}`` to the RLinf setup script to install either GR00T generation. The two
  releases share the ``gr00t`` package name, so one environment holds one generation at a time.

Changed
^^^^^^^

* Changed the RLinf pin installed by ``setup_rlinf.py`` from the ``0.2.0dev2`` PyPI release to a git
  commit that dispatches on ``actor.model.model_type``, so one RLinf serves GR00T N1.5 and N1.7.
  Re-run the setup script to upgrade an existing environment.

Fixed
^^^^^

* Fixed ``--checkpoint`` being silently ignored by the RLinf play entrypoint. It was written to
  ``runner.eval_policy_path``, which RLinf does not read; the RL-finetuned weights are now loaded
  through the RLinf extension instead. A ``global_step_<N>`` directory is resolved to the
  ``full_weights.pt`` it contains.
* Fixed RLinf evaluation on RLinf 0.3, whose rollout worker reads ``rollout.model`` directly instead
  of copying ``actor.model``: the train and play entrypoints now fill every ``actor.model`` key the YAML
  leaves out of ``rollout.model`` (``model_type``, ``rl_head_config``, ...), so task configs need not
  mirror them by hand.
* Changed ``--checkpoint`` for GR00T N1.7 tasks to hand the resolved ``full_weights.pt`` to RLinf's own
  ``runner.ckpt_path`` hook, since RLinf builds N1.7 models natively and the extension's weight overlay
  only applies to N1.5. Hugging Face-format exports of an RL run load through ``--model_path`` instead.
* Fixed relative ``model_path`` values in RLinf configs resolving against the Ray worker's working
  directory instead of the launcher's. Both the train and play entrypoints now make the path absolute
  before the config reaches the workers.
* Fixed ``--checkpoint`` never resuming RLinf training. The train entrypoint set ``runner.resume_dir``
  to the directory holding ``full_weights.pt``, while RLinf appends ``actor`` to that path and reads the
  step count out of its name; resuming therefore failed on its assertion. ``_resolve_rlinf_resume_dir``
  now returns the enclosing ``global_step_<N>`` directory, whichever of the three accepted forms
  ``--checkpoint`` was given.
