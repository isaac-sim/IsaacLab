* **Breaking:** Removed the ``Se2/Se3Keyboard``, ``Se2/Se3Gamepad`` and ``Se2/Se3SpaceMouse`` input
  implementations and their HID helpers. Deprecated configuration records remained available and
  were translated to Isaac Capture configs before launch. Direct constructors were replaced by the
  :mod:`isaaclab_teleop` pipeline builders and :func:`~isaaclab_teleop.create_isaac_teleop_device`;
  see :ref:`isaac-teleop-keyboard` for migration and standalone session setup.
