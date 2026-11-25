isaaclab_fovi.sensors
=====================

.. automodule:: isaaclab_fovi.sensors

  .. rubric:: Classes

  .. autosummary::

    FoveatedCamera
    FoveatedCameraCfg
    FoveatedNewtonCamera
    FoveatedRTXCamera
    ViewerGazeControls


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

Viewer Controls
---------------

Each foveated camera adds these controls to every Newton viewer window, below the Camera Source selector while it
is the window's camera source.

.. autoclass:: ViewerGazeControls
    :members: cmf_a_values
