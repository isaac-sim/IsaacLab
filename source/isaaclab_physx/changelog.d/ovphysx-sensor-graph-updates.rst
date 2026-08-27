Changed
^^^^^^^

* Changed the PhysX contact sensor and ray caster to build their CUDA-graph
  capture and replay on :class:`~isaaclab.utils.warp.CapturedKernelUpdate`
  instead of duplicating the capture logic in each sensor. Behavior is
  unchanged. The private ``_use_graph`` and ``_compute_graph`` attributes were
  replaced by a single ``_update_graph`` helper instance; use
  ``sensor._update_graph.enabled`` in place of ``sensor._use_graph``.
