* Changed remote asset retrieval in :mod:`isaaclab.utils.assets` from ``omni.client`` to
  `OVStorage <https://github.com/NVIDIA-Omniverse/ovstorage>`__, which replaces the ``omniverseclient``
  dependency. Where OVStorage is not installed, such as macOS, which it publishes no wheels for, public HTTP(S)
  assets are read with the standard library.
* Changed the asset region profiles to route Isaac Lab's own asset reads through the profile's CDN.
  :func:`~isaaclab.utils.assets.configure_storage_profile` still configures ``omni.client`` for Kit, and is
  skipped when ``omni.client`` is not installed.
