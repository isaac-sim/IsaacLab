* Deprecated the ``Se2/Se3KeyboardCfg``, ``Se2/Se3GamepadCfg`` and ``Se2/Se3SpaceMouseCfg``
  configuration records in Isaac Lab 3.2. Existing declarations continue to translate into
  Isaac Capture configs before device creation. Use the :mod:`isaaclab_teleop` pipeline builders
  for new configurations; see :ref:`isaac-teleop-keyboard` for migration.
