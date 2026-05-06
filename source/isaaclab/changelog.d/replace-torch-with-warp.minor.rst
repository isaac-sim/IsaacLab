Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.observations.image` (and the equivalent
  ``image()`` helper in the Franka stack ``stack_ik_rel_blueprint`` config) so
  that Torch tensor operations are applied via :func:`warp.to_torch` rather than
  invoked directly on the new ``wp.array`` camera outputs.
* Fixed downstream consumers (camera tutorials, ``demos/sensors/cameras.py``,
  ``benchmarks/benchmark_cameras.py``, dexsuite ``vision_camera`` observation,
  visualizer integration test, ``save_camera_output`` how-to and the camera
  overview docs) to lift ``wp.array`` camera fields to torch tensors via
  :func:`warp.to_torch` before performing Torch operations.
* Fixed the camera and test suites to use ``wp.array`` dtypes
  (``wp.uint8``, ``wp.float32``, ``wp.int32``) and :func:`warp.to_torch` views
  in assertions on ``CameraData.output``, ``CameraData.intrinsic_matrices``,
  ``CameraData.pos_w``/``quat_w_*`` and ``Camera.frame``.
* Fixed :class:`~isaaclab.renderers.NewtonWarpRenderer` to populate the ``rgb``
  output buffer when both ``rgb`` and ``rgba`` are requested, restoring the
  legacy "rgb mirrors rgba" behavior that broke when ``rgb`` and ``rgba``
  became independent ``wp.array`` allocations.

Changed
^^^^^^^

* Tightened :class:`~isaaclab.renderers.RenderBufferSpec` ``dtype`` annotation
  from ``Any`` to ``type`` to document that all renderers must publish Warp
  scalar dtype classes (e.g. ``warp.uint8``).
* Removed the transitional ``torch.dtype → wp.dtype`` shim in
  :meth:`~isaaclab.sensors.camera.CameraData.allocate` now that all in-tree
  renderers publish ``wp`` dtypes via :class:`~isaaclab.renderers.RenderBufferSpec`.
* Documented the transitional Torch input fallback on
  :func:`~isaaclab.utils.math.convert_camera_frame_orientation_convention` and
  consolidated the redundant ``wp ↔ torch`` round-trips in
  :meth:`~isaaclab.sensors.camera.Camera.set_world_poses` and
  :meth:`~isaaclab.sensors.camera.Camera.set_world_poses_from_view`.
* Changed :class:`~isaaclab.sensors.camera.CameraData` array fields and
  :attr:`~isaaclab.sensors.camera.CameraData.output` buffers to expose
  ``wp.array`` values instead of :class:`torch.Tensor` values. Use
  :func:`warp.to_torch` where Torch tensor operations are required.
* Changed :class:`~isaaclab.sensors.camera.Camera` pose, intrinsic, and frame
  array APIs to accept or return ``wp.array`` values instead of
  :class:`torch.Tensor` values. Existing Torch inputs are still accepted during
  the transition; prefer ``wp.array`` at public call sites.
* Changed :class:`~isaaclab.renderers.BaseRenderer` output and camera-update
  APIs to exchange ``wp.array`` buffers with camera sensors.
* Changed :func:`~isaaclab.utils.math.convert_camera_frame_orientation_convention`
  to accept and return ``wp.array`` quaternion arrays. Use :func:`warp.to_torch`
  where Torch tensor operations are required.
