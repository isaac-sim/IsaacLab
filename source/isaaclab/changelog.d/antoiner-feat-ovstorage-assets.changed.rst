* **Breaking:** Changed remote asset access in :mod:`isaaclab.utils.assets` from ``omni.client`` to
  `OVStorage <https://github.com/NVIDIA-Omniverse/ovstorage>`__, which replaces the ``omniverseclient``
  dependency. The asset region profile's object-storage endpoint is now read through its CDN.
