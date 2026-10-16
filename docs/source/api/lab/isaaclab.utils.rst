isaaclab.utils
==============

.. automodule:: isaaclab.utils

   .. Rubric:: Submodules

   .. autosummary::

      io
      array
      assets
      buffers
      datasets
      dict
      interpolation
      logger
      math
      mesh
      tensor_processors
      noise
      seed
      sensors
      string
      timer
      types
      version
      warp

   .. Rubric:: Functions

   .. autosummary::

      configclass
      instantiate
      clone
      replace
      validate
      to_dict
      update_from_dict

Configuration class
~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.configclass
   :members:
   :show-inheritance:

IO operations
~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.io
   :members:
   :imported-members:
   :show-inheritance:

Array operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.array
   :members:
   :show-inheritance:

Asset operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.assets
   :members:
   :show-inheritance:

Buffer operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.buffers
   :members:
   :imported-members:
   :inherited-members:
   :show-inheritance:

Datasets operations
~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.datasets
   :members:
   :show-inheritance:
   :exclude-members: __init__, func

Dictionary operations
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.dict
   :members:
   :show-inheritance:

Interpolation operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.interpolation
   :members:
   :imported-members:
   :inherited-members:
   :show-inheritance:

Logger operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.logger
   :members:
   :show-inheritance:

Math operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.math
   :members:
   :inherited-members:
   :show-inheritance:

Mesh operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.mesh
   :members:
   :imported-members:
   :show-inheritance:

Tensor processors
~~~~~~~~~~~~~~~~~

The processor interfaces live together in :mod:`isaaclab.utils.tensor_processors`:

* ``TensorProcessor``: tensor to tensor.
* ``DictTensorProcessor``: dictionary of named tensors to dictionary of named tensors.

Dictionary tensors use ``ProxyArray`` for Warp and Torch access without copying. Processors own
any persistent outputs and state. ``DictTensorProcessorCfg`` declares input/output keys so consumers
can resolve dependencies before allocating buffers. Composition preserves untouched keys.

Migration: import ``TensorProcessor`` and ``TensorProcessorCfg`` from
``isaaclab.utils.tensor_processors`` in place of ``ModifierBase`` and ``ModifierCfg`` from
``isaaclab.utils.modifiers``. Replace ``ObservationTermCfg.modifiers`` with
``ObservationTermCfg.tensor_processors``. The processing order and tensor shapes are unchanged.
The old imports and configuration field remain available, with one shared deprecation warning on first use.
Do not set both fields.

.. automodule:: isaaclab.utils.tensor_processors
   :members:
   :imported-members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, func

Noise operations
~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.noise
   :members:
   :imported-members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, func

Seed operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.seed
   :members:
   :show-inheritance:

Sensor operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.sensors
   :members:
   :show-inheritance:

String operations
~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.string
   :members:
   :show-inheritance:

Image operations
~~~~~~~~~~~~~~~~

.. autofunction:: isaaclab.utils.images.compose_image

Timer operations
~~~~~~~~~~~~~~~~

By default, :class:`~isaaclab.utils.timer.Timer` waits for all GPU devices when it stops,
preserving its existing behavior. Work queued before the timer starts can therefore contribute
to the measured duration. Choose synchronization boundaries explicitly when profiling:

.. code-block:: python

   # CPU wall time without waiting for GPU work.
   with Timer(synchronize="none"):
       preprocess()

   # Finish earlier work before starting, then wait for this device before stopping.
   with Timer(synchronize="both", device="cuda:0"):
       step()

``synchronize="stop"`` selects the default stop-only boundary. ``device=None`` synchronizes all
devices; a device name restricts synchronization to that device, including all of its streams.
These measurements include CPU overhead and concurrent work on the selected device. They are
wall-clock measurements, not CUDA event measurements. :attr:`~isaaclab.utils.timer.Timer.time_elapsed`
does not synchronize pending work.

.. automodule:: isaaclab.utils.timer
   :members:
   :show-inheritance:

Type operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.types
   :members:
   :show-inheritance:

Version operations
~~~~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.version
   :members:
   :show-inheritance:

Warp operations
~~~~~~~~~~~~~~~

.. automodule:: isaaclab.utils.warp
   :members:
   :imported-members:
   :show-inheritance:

.. rubric:: Particle sampling functions

.. autosummary::

   sample_particles_in_mesh
   sample_particles_in_cavity

Warp Fabric kernels
^^^^^^^^^^^^^^^^^^^

Warp kernels for reading and writing Fabric ``Matrix4d`` attributes
(``omni:fabric:worldMatrix`` / ``omni:fabric:localMatrix``) via
:class:`wp.fabricarray` and :class:`wp.indexedfabricarray`. Used by
:class:`~isaaclab_physx.sim.views.FabricFrameView` to keep child world and
local matrices consistent without round-tripping through USD.

.. automodule:: isaaclab.utils.warp.fabric
   :members:
   :show-inheritance:
