isaaclab.renderers
==================

.. automodule:: isaaclab.renderers

  .. rubric:: Classes

  .. autosummary::

    BaseRenderer
    RendererCfg

  .. rubric:: Functions

  .. autosummary::

    async_rendering_enabled_from_env
    resolve_async_rendering_enabled
    warn_unsupported_async_rendering

Base Renderer
-------------

.. autoclass:: BaseRenderer
  :members:
  :show-inheritance:
  :exclude-members: __init__

Renderer Configuration
-----------------------

.. autoclass:: RendererCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__


Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.renderers` API.

.. currentmodule:: isaaclab.renderers

.. autosummary::
   :nosignatures:

   CameraRenderSpec
   RenderBufferKind
   RenderBufferSpec
   RenderContext

.. autoclass:: CameraRenderSpec
   :show-inheritance:

.. autoclass:: RenderBufferKind
   :show-inheritance:

.. autoclass:: RenderBufferSpec
   :show-inheritance:

.. autoclass:: RenderContext
   :show-inheritance:

Asynchronous Rendering
----------------------

:attr:`RendererCfg.async_rendering` trades one frame of camera latency for pipelined rendering, and
:data:`ASYNC_RENDERING_ENV_VAR` overrides it process-wide. Renderers resolve it through these
helpers; only the OVRTX renderer implements the pipelined path.

.. autodata:: ASYNC_RENDERING_ENV_VAR

.. autofunction:: resolve_async_rendering_enabled

.. autofunction:: async_rendering_enabled_from_env

.. autofunction:: warn_unsupported_async_rendering
