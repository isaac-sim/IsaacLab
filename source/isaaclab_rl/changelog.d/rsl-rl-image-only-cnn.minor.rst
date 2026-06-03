Added
^^^^^

* Added :class:`~isaaclab_rl.rsl_rl.models.CNNModel`, a drop-in extension of rsl-rl's ``CNNModel`` that
  also supports image-only policies (observation sets with no 1D groups). rsl-rl's model raises
  ``ValueError: torch.cat(): expected a non-empty list of Tensors`` in that case.

Changed
^^^^^^^

* Changed :attr:`~isaaclab_rl.rsl_rl.RslRlCNNModelCfg.class_name` to default to
  :class:`~isaaclab_rl.rsl_rl.models.CNNModel` so CNN policies support image-only observations out of
  the box. The new model is identical to rsl-rl's ``CNNModel`` when 1D observation groups are present;
  set ``class_name="CNNModel"`` to restore the previous behavior.
