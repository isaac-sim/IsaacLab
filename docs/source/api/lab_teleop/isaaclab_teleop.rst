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
