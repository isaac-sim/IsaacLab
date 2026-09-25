Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.warp.math_ops.transform_to_vec_quat` raising a Warp ``RuntimeError`` instead of its
  documented ``ValueError`` for 4D transform arrays.
