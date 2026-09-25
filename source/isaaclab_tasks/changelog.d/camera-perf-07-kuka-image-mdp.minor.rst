Changed
^^^^^^^

* Changed the Kuka Allegro Lift camera observations to use the shared
  :func:`~isaaclab.envs.mdp.observations.image` term and to emit raw camera images, uint8 for
  color. The spatial-softmax policy model now applies the fixed-range normalization, including in
  exported JIT and ONNX policies, which cuts rollout image memory 4x. Existing checkpoints load
  unchanged; policies exported before this change expect normalized images.
* Disabled the RSL-RL per-step NaN check for the Kuka Allegro camera runners.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab_tasks.core.lift.mdp.vision_camera``. Use
  :func:`~isaaclab.envs.mdp.observations.image` with ``data_type=None``, ``normalize=False``, and
  ``permute=True``, and normalize the raw images in the policy, as the Kuka Allegro camera tasks do.
