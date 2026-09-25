Changed
^^^^^^^

* Changed the Kuka Allegro Lift camera observations to use the shared
  :func:`~isaaclab.envs.mdp.observations.image` term and to emit raw camera images, uint8 for
  color. The spatial-softmax policy model now applies the fixed-range normalization, including in
  exported JIT and ONNX policies, which cuts rollout image memory 4x. Existing checkpoints load
  unchanged; policies exported before this change expect normalized images.
* Changed the default of ``visualize`` in ``isaaclab_tasks.core.lift.mdp.object_point_cloud_b`` to
  False. Drawing every point each step was costly and, with RTX rendering, put the markers into
  camera images. Pass ``visualize=True`` to draw them.
* Disabled the RSL-RL per-step NaN check for the Kuka Allegro camera runners.

Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab_tasks.core.lift.mdp.vision_camera``. Use
  :func:`~isaaclab.envs.mdp.observations.image` with ``data_type=None``, ``normalize=False``, and
  ``permute=True``, and normalize the raw images in the policy, as the Kuka Allegro camera tasks do.

Fixed
^^^^^

* Fixed the Cartpole camera observations reading the sensor's ``ProxyArray`` directly, which left
  colorized semantic segmentation unscaled.
