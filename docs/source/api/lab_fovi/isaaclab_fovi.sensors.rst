isaaclab_fovi.sensors
=====================

.. automodule:: isaaclab_fovi.sensors

  .. rubric:: Classes

  .. autosummary::

    FoveatedCamera
    FoveatedCameraCfg
    FoveatedNewtonCamera
    FoveatedRTXCamera


Foveated Camera
---------------

.. autoclass:: FoveatedCamera
    :members:
    :show-inheritance:

.. autoclass:: FoveatedCameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Renderer Implementations
------------------------

Constructing :class:`FoveatedCamera` returns one of the following implementations, selected by the renderer in
:attr:`FoveatedCameraCfg.renderer_cfg`.

.. autoclass:: FoveatedNewtonCamera
    :members:
    :show-inheritance:

.. autoclass:: FoveatedRTXCamera
    :members:
    :show-inheritance:
