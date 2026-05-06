Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to populate
  the ``rgb`` output buffer when both ``rgb`` and ``rgba`` are requested.
  Without the fix, ``CameraData.output["rgb"]`` was left zero-filled because
  the renderer skipped ``rgb`` in :meth:`read_output` after ``rgb`` and
  ``rgba`` became independent ``wp.array`` allocations.


Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to consume
  ``wp.array`` camera output buffers and camera state arrays from
  :class:`~isaaclab.renderers.BaseRenderer`. Use :func:`warp.to_torch` on
  ``camera.data.output`` entries if Torch tensor operations are required.
* Updated Newton PVA debug visualization to convert camera-convention
  orientation outputs with :func:`warp.to_torch`.
