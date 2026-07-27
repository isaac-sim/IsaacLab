Changed
^^^^^^^

* Changed the experience-file fallback for the asset root to read
  ``persistent.isaac.asset_root.default`` before the legacy
  ``persistent.isaac.asset_root.cloud``, matching the setting that Isaac Sim resolves.
  Redirecting Isaac Lab assets by editing only ``persistent.isaac.asset_root.cloud`` in an
  experience file therefore no longer has any effect, because the shipped experiences leave
  ``persistent.isaac.asset_root.default`` pointing at the cloud. Set the
  ``ISAACSIM_ASSET_ROOT`` environment variable instead, or edit
  ``persistent.isaac.asset_root.default``.

* Changed the ``isaaclab.python`` and ``isaaclab.python.headless`` experiences to load
  ``isaacsim.storage.native``, so the Isaac Sim asset-root APIs apply
  ``ISAACSIM_ASSET_ROOT`` as well.

Fixed
^^^^^

* Fixed the asset root ignoring the documented ``ISAACSIM_ASSET_ROOT`` environment
  variable, which prevented local, self-hosted, and air-gapped asset roots from being
  used with :attr:`~isaaclab.utils.assets.NUCLEUS_ASSET_ROOT_DIR` and the paths derived
  from it.
