Fixed
^^^^^

* Fixed the JIT and ONNX export of image-only :class:`~isaaclab_rl.rsl_rl.models.CNNModel`
  policies. The exported models no longer require feeding a zero-width 1D observation
  input (``obs``); they now only take the 2D observation groups as inputs.
