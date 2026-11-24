* Updated the assemble-trocar robot, scene and instrument assets, reset poses and camera configuration.
  Extract ``assemble-trocar-assets.zip`` into ``/workspace/assets`` or set ``ASSET_LOCAL`` to
  the directory or HTTPS URL containing ``Robots/`` and ``Props/``.
* Updated the task to use 320-by-240 RGB observations, 30 Hz control and absolute joint-position actions.
* Updated the instrument lift criterion to use center-of-mass height for the revised USD geometry.
* Updated the RLinf recipe to use the 100k SFT checkpoint directory
  ``.pretrained_checkpoints/rlinf/Assemble_Trocar_SFT_100k`` and the instruction
  ``install trocar from tray`` for training and evaluation. Supply the 100k checkpoint locally
  or override ``--model_path``; the previously published checkpoint does not contain these weights.
