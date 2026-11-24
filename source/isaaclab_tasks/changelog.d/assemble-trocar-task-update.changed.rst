* Updated the assemble-trocar robot, scene and instrument assets, reset poses and camera configuration.
  Extract ``assemble-trocar-assets.zip`` into ``/workspace/assets`` or set ``ASSET_LOCAL`` to
  the directory or HTTPS URL containing ``Robots/`` and ``Props/``.
* Updated the task to use 320-by-240 RGB observations, 30 Hz control and absolute joint-position actions.
* Updated the instrument lift criterion to use center-of-mass height for the revised USD geometry.
