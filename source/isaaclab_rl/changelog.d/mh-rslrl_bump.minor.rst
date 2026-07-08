Changed
^^^^^^^

* Bumped the ``rsl-rl-lib`` dependency to ``5.4.1``, which natively supports image-only policies
  (observation sets with no 1D groups).
* Changed :attr:`~isaaclab_rl.rsl_rl.RslRlCNNModelCfg.class_name` to default to rsl-rl's
  ``CNNModel`` now that it supports image-only observations out of the box.

Removed
^^^^^^^

* Removed the Isaac Lab ``CNNModel`` override of rsl-rl's ``CNNModel`` that previously added
  image-only observation support. Use rsl-rl's ``CNNModel`` (the new default of
  :attr:`~isaaclab_rl.rsl_rl.RslRlCNNModelCfg.class_name`) instead.
