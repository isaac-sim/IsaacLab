Added
^^^^^

* Added a temporary ``warp.torch`` compatibility shim at
  :mod:`isaaclab_mimic` import time so that cuRobo (NVlabs/curobo) keeps
  working with ``warp-lang>=1.13``, which dropped the ``warp.torch``
  submodule in favour of top-level ``warp.*`` (e.g.
  ``wp.torch.device_from_torch`` → ``wp.device_from_torch``). cuRobo's
  pinned commit and ``main`` still call ``wp.torch.*`` and raise
  ``AttributeError: module 'warp' has no attribute 'torch'`` at
  :meth:`MotionGenConfig.load_from_robot_config` time. The shim
  reconstructs ``warp.torch`` as a thin forwarding module and is a
  no-op once warp re-introduces the namespace or cuRobo migrates.
  Remove this shim once the cuRobo pin in ``docker/Dockerfile.curobo``
  is bumped to a commit that uses the top-level ``wp.*`` API directly.
