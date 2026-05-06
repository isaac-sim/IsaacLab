Fixed
^^^^^

* Fixed import of ``omni.physics.tensors.api`` across :mod:`isaaclab_physx`
  assets and sensors. The upstream ``omni.physics.tensors`` package no longer
  exposes ``api`` as a top-level submodule (only as a re-exported attribute via
  ``omni.physics.tensors.impl.api``), so ``import omni.physics.tensors.api as physx``
  raised ``ModuleNotFoundError`` at runtime. Replaced the dotted-form imports
  with ``from omni.physics.tensors import api as physx``, which resolves
  through the package's attribute table and works against both old and new
  layouts.
