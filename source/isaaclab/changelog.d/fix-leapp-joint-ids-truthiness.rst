Fixed
^^^^^

* Fixed :class:`~isaaclab.utils.leapp.export_annotator.ExportPatcher` failing to export PD
  gains for action terms whose ``joint_ids`` is a tensor or a ``wp.array``, which raised
  ``RuntimeError: Boolean value of Tensor with more than one value is ambiguous`` and
  ``RuntimeError: Item indexing is not supported on wp.array objects``. Joint names are now
  also resolved for ``wp.array`` selections instead of being silently dropped.
