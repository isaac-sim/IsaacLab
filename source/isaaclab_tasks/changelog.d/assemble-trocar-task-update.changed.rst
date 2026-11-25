* **Breaking:** Updated the existing assemble-trocar task's robot, scene and instrument assets,
  reset poses and camera presets for the matching 100k SFT policy.
  Extract ``assemble-trocar-assets.zip`` into ``/workspace/assets`` or set ``ASSET_LOCAL`` to
  the directory or HTTPS URL containing ``Robots/`` and ``Props/``.
  Retain the earlier Isaac Lab revision, assets and recipe to reproduce older policies.
* Updated the task to use 320-by-240 RGB observations, 30 Hz control, 25-second episodes
  and absolute joint-position actions.
* Updated the lift stage and reward defaults from a 0.05 m to a 0.08 m threshold above the table,
  using center-of-mass height for the first trocar and root height for the second.
* Updated the RLinf recipe to use the 100k SFT checkpoint directory
  ``.pretrained_checkpoints/rlinf/Assemble_Trocar_SFT_100k`` and the instruction
  ``install trocar from tray`` for training and evaluation. Use ``--model_path`` to select another checkpoint.
