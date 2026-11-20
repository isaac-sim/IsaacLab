* Added ``isaaclab_neural``, a world-model physics backend with a virtual articulation sharing device joint-state
  and target buffers, a pure ``WorldModel`` interface, and a renderer for named RGB camera views.
* Added device-side reset-record sampling with train/evaluation splits and environment grouping. The explicit
  ``reset_from_records`` event restores recorded state and images while preserving configured default joint positions.
* Added per-view frame buffers and independent camera readers, chunk-aligned termination and decimation helpers,
  and initialization checks for model environment limits, reset joint counts, and single-articulation binding.
