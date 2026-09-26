Changed
^^^^^^^

* **Breaking:** Replaced the default ``Isaac-Velocity-Rough-G1`` and ``Isaac-Velocity-Flat-G1`` tasks
  with 29-body-action G1 walking configurations and passive finger joints. Both tasks
  now expose 286 policy observations, including the height scan. RSL-RL training
  defaults were set to 6000 iterations with left-right augmentation for Rough and
  1500 without augmentation for Flat. Rough used an arm deviation weight of -0.8
  and a terrain-relative pelvis height target of 0.75 m. Flat retained lateral
  commands and used completed swing/stance duration variance and flight penalties.

  Existing checkpoints trained with the previous default tasks are incompatible with
  the new action/observation interfaces. Retrain with the new defaults, or use the
  previous task definitions and asset when playing those checkpoints.
