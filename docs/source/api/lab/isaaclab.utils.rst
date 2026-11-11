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
      noise
      seed
      sensors
      string
      tensor_processors
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

Tensor processors
~~~~~~~~~~~~~~~~~

Use :mod:`isaaclab.utils.tensor_processors` to transform tensors or nested tensor collections.
:class:`~isaaclab.utils.tensor_processors.TensorProcessor` and
:class:`~isaaclab.utils.tensor_processors.TensorProcessorCfg` support ``torch.Tensor``,
:class:`~isaaclab.utils.warp.ProxyArray`, and recursively nested lists, tuples, and dictionaries
through ``TensorProcessorData``. Each implementation defines the structures it accepts and returns;
the base class does not traverse collections automatically.

Set ``cfg.func`` to a function, a processor class, or a lazily resolved import string. Functions
receive ``cfg.params`` as keyword arguments; classes read settings from their configuration.
Optional ``inputs`` and ``outputs`` lists describe named keys but do not select or route data.
For example, a subclass can stack a nonempty list of dictionaries whose ``position`` tensors
share a shape, dtype, and device:

.. code-block:: python

   import torch
   from isaaclab.utils.tensor_processors import TensorProcessor, TensorProcessorCfg


   class StackPositions(TensorProcessor):
       def __init__(self, cfg: TensorProcessorCfg, data: list[dict[str, torch.Tensor]]) -> None:
           super().__init__(cfg)
           first = data[0]["position"]
           self.output = torch.empty((len(data), *first.shape), dtype=first.dtype, device=first.device)

       def __call__(self, data: list[dict[str, torch.Tensor]]) -> torch.Tensor:
           return torch.stack([item["position"] for item in data], out=self.output)


   data = [{"position": torch.zeros(3)}, {"position": torch.ones(3)}]
   cfg = TensorProcessorCfg(func=StackPositions, inputs=["position"])
   processor = cfg.func(cfg, data)
   stacked_positions = processor(data)

Construction uses input metadata without reading values or advancing state. Subsequent calls use
the same number of input dictionaries and tensor shapes. Inputs are read-only and owned by the
caller. This subclass owns its persistent output, which remains valid until the next call; copy
results that must be retained. ``reset`` and ``close`` do nothing unless overridden. Processors
using Warp launch work on the caller's current Warp stream.

Observation terms continue to use shape-preserving single-tensor processors through
:attr:`~isaaclab.managers.ObservationTermCfg.tensor_processors`. The observation manager constructs
class processors with ``cfg.func(cfg, data_dim, device)``. Built-in filters retain this constructor
and their single-tensor behavior; structured processors are constructed directly as above.

Migration: import ``TensorProcessor`` and ``TensorProcessorCfg`` from
``isaaclab.utils.tensor_processors`` in place of ``ModifierBase`` and ``ModifierCfg`` from
``isaaclab.utils.modifiers``. Replace ``ObservationTermCfg.modifiers`` with
``ObservationTermCfg.tensor_processors``. The processing order and tensor shapes are unchanged.
Old imports remain supported and emit ``DeprecationWarning`` with migration instructions.

.. automodule:: isaaclab.utils.tensor_processors
   :members:
   :imported-members:
   :special-members: __call__
   :inherited-members:
   :show-inheritance:
   :exclude-members: __init__, func

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

Warp device math
^^^^^^^^^^^^^^^^

``isaaclab.utils.warp.math`` provides device-callable functions and generated array kernels for
quaternion, rigid-body, interval, and Jacobian math. Quaternions use ``xyzw`` order, and spatial
vectors contain linear components before angular components. Tensor operations, broadcasting,
and Torch autograd remain in ``isaaclab.utils.math``.

.. automodule:: isaaclab.utils.warp.math
   :members:

Warp memory operations
^^^^^^^^^^^^^^^^^^^^^^

Use Warp's native array operations for ordinary data movement. The shared
``isaaclab.utils.warp.memory`` module supplies selector normalization, component views,
boolean-masked updates and fused packing/splitting. Callers own attribute frequencies:
resolve them into shapes, strides and selections. The module has no body, joint or
coordinate-domain registry. Physical equations and solver mappings remain in their owners.

For example::

    import warp as wp
    from isaaclab.utils.warp import memory

    rows = memory.resolve_1d_indices(rows)
    columns = memory.resolve_1d_indices(columns)
    selected = wp.indexedarray(target, [rows, columns])
    wp.copy(selected, source)  # compact source matching selected.shape
    selected.fill_(0.0)
    wp.copy(selected, wp.indexedarray(full_source, [rows, columns]))

    # Dynamic boolean masks stay on the device without compaction.
    wp.launch(memory.make_masked_fill(wp.float32, 2), target.shape,
              [0.0, row_mask, column_mask, target])

    # Pack typed components into existing storage in one tiled launch.
    memory.concat(first, second, result)

``resolve_1d_indices`` preserves the order of valid nonnegative indices, returning
contiguous int32 storage as required by Warp indexed views. It reuses contiguous int32
arrays; strided/int64 arrays need a copy or cast. Keep normalized indices alive with the
view. Warm conversions before CUDA graph capture; capture-time allocation requires the
Warp memory pool. Source and destination data may remain strided. Overlapping destination
indices require identical values. Simulator row-ID bookkeeping stays with the backend.

``split_views`` returns component aliases without allocating storage. Prefer scalar slices
for native copies of component data. Retained fused packing/splitting kernels avoid extra
launches and can use better memory access patterns than multiple component copies.
``concat`` owns scalar views and the tiled launch grid. It accepts rank-1 or rank-2
composite arrays, preserves strides and uses the same specialization for contiguous
arrays of either rank. It allocates no array storage. Input and output storage must
not overlap. With ``record_cmd=True`` it returns a native Warp launch command; reuse
that command for stable eager CUDA inputs to avoid rebuilding views on each call.

For custom operations, cached factories return concrete kernels in isolated modules.
Only requested layouts are constructed. Python caches kernel objects; Warp separately
caches compilation. Warm the execution path before capture. Later specializations must
not invalidate kernels already compiled or captured.

``resolve_1d_mask`` owns selection normalization for sensors and experimental environments.
Same-device signed integer arrays, boolean masks and slices are capturable; host sequences
and device transfers are eager-only; temporary GPU index uploads synchronize their stream
before releasing storage. It preserves integer widths, negative indices and array
strides. Invalid device indices are ignored without synchronization. Scratch storage is reused,
so callers must not retain a scratch result across another resolution using the same buffer.

.. automodule:: isaaclab.utils.warp.memory
   :members:

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
