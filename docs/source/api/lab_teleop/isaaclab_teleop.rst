.. _isaaclab_teleop-api:

isaaclab_teleop
===============

.. automodule:: isaaclab_teleop

  .. rubric:: Classes

  .. autosummary::

    IsaacTeleopCfg
    IsaacTeleopDevice
    XrCameraFeedCfg
    XrCameraFeedLayoutCfg
    XrCameraFeedSession
    HapticFeedbackCfg
    ControllerHapticFeedbackCfg
    GloveHapticFeedbackCfg
    HapticFeedbackReceiver
    HapticFeedbackDriver
    XrCfg
    XrAnchorRotationMode
    XrAnchorSynchronizer

  .. rubric:: Functions

  .. autosummary::

    create_isaac_teleop_device
    create_haptic_feedback_driver
    remove_camera_configs
    se2_gamepad_teleop_cfg
    se2_keyboard_teleop_cfg
    se2_spacemouse_teleop_cfg
    se3_gamepad_teleop_cfg
    se3_keyboard_teleop_cfg
    se3_spacemouse_teleop_cfg

Configuration
-------------

.. autoclass:: IsaacTeleopCfg
    :members:

.. autoclass:: XrCfg
    :members:

.. autoclass:: XrAnchorRotationMode
    :members:

XR Camera Feedback
------------------

.. autoclass:: XrCameraFeedCfg
    :members:

.. autoclass:: XrCameraFeedLayoutCfg
    :members:

.. autoclass:: XrCameraFeedSession
    :members:

Device
------

.. autoclass:: IsaacTeleopDevice
    :members:
    :show-inheritance:

.. autofunction:: create_isaac_teleop_device

Keyboard
--------

.. autofunction:: se2_keyboard_teleop_cfg

.. autofunction:: se3_keyboard_teleop_cfg

Gamepad
-------

.. autofunction:: se2_gamepad_teleop_cfg

.. autofunction:: se3_gamepad_teleop_cfg

SpaceMouse
----------

.. autofunction:: se2_spacemouse_teleop_cfg

.. autofunction:: se3_spacemouse_teleop_cfg

Haptic Feedback
---------------

.. autoclass:: HapticFeedbackCfg
    :members:

.. autoclass:: ControllerHapticFeedbackCfg
    :members:
    :show-inheritance:

.. autoclass:: GloveHapticFeedbackCfg
    :members:
    :show-inheritance:

.. autoclass:: HapticFeedbackReceiver
    :members:

.. autoclass:: HapticFeedbackDriver
    :members:

.. autofunction:: create_haptic_feedback_driver

XR Anchor
---------

.. autoclass:: XrAnchorSynchronizer
    :members:

.. autofunction:: remove_camera_configs

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab_teleop` API.

.. currentmodule:: isaaclab_teleop

.. autosummary::
   :nosignatures:

   ControlEvents
   SupportsControlEvents
   SystemCheckItem
   SystemCheckResult

.. autoclass:: ControlEvents
   :show-inheritance:

.. autoclass:: SupportsControlEvents
   :show-inheritance:

.. autoclass:: SystemCheckItem
   :show-inheritance:

.. autoclass:: SystemCheckResult
   :show-inheritance:

Entrypoints
-----------

.. automodule:: isaaclab_teleop.entrypoints

.. autofunction:: isaaclab_teleop.entrypoints.run_cli

.. autofunction:: isaaclab_teleop.entrypoints.run_teleop_se3_agent_cli

.. autofunction:: isaaclab_teleop.entrypoints.run_record_demos_cli

.. autofunction:: isaaclab_teleop.entrypoints.run_replay_demos_cli

.. autofunction:: isaaclab_teleop.entrypoints.run_teleop_replay_agent_cli
