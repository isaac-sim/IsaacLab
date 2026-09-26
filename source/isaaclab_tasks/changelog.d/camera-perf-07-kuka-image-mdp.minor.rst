Changed
^^^^^^^

* Changed the Kuka Allegro Lift camera observations to use the shared
  :class:`~isaaclab.envs.mdp.observations.image_rgb`,
  :class:`~isaaclab.envs.mdp.observations.image_depth`, and
  :class:`~isaaclab.envs.mdp.observations.image_segmentation` terms and to emit raw camera images, uint8 for
  color. The spatial-softmax policy model now applies the fixed-range normalization, including in
  exported JIT and ONNX policies, which cuts rollout image memory 4x. Existing checkpoints load
  unchanged; policies exported before this change expect normalized images.
* Disabled the RSL-RL per-step NaN check for the Kuka Allegro camera runners.
