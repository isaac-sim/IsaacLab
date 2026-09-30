isaaclab.sensors.camera
=======================

.. automodule:: isaaclab.sensors.camera

See :class:`~isaaclab.sensors.Camera`, :class:`~isaaclab.sensors.CameraCfg`, and
:class:`~isaaclab.sensors.CameraData` for the camera API, and :ref:`camera-post-processing`
for configuring image processing through observation terms.

``CameraCfg.isp_cfg`` and the following discovery enum remain available during migration.
The deprecated camera adapter preserves processed camera outputs; new observation chains
use :class:`~isaaclab_ppisp.PpispProcessorCfg` and :class:`~isaaclab_ppisp.PpispDiscoveryMode`.

.. autoclass:: isaaclab.sensors.camera.CameraISPMode
   :members:
   :undoc-members:
