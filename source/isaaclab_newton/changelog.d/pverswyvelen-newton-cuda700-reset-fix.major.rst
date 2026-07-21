Fixed
^^^^^

* Fixed a CUDA ``700`` (illegal memory access) that could occur on the first
  simulation step after a hard reset (:meth:`NewtonManager.reset` with
  ``soft=False``). On a hard reset the Newton :class:`Model` is re-created,
  which reallocates its device arrays, but the collision pipeline still held
  references to the old model. Once the old model's device buffers were freed
  and reused (which happens under GPU memory pressure between resets), those
  dangling references caused an illegal memory access on the first ``step()``
  after the reset (typically surfacing in ``compute_shape_aabbs`` or
  ``narrow_phase_kernel_gjk_mpr``). The reset now clears the cached collision
  pipeline and contacts so a fresh pipeline is rebuilt against the re-finalized
  model, and invalidates any previously-captured CUDA graph. The graph is then
  re-captured by :meth:`initialize_solver` exactly as on first initialization.
  This restores correct behavior with CUDA graph capture enabled.
