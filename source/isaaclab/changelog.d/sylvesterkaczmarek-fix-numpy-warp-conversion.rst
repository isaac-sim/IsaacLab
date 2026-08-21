Fixed
^^^^^

* Fixed ``convert_dict_to_backend(..., backend="warp")`` rejecting NumPy arrays because the
  conversion registry used the ``np.array`` constructor instead of the ``np.ndarray`` type.
