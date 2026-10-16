.. _concepts_sensors_camera:
.. _overview_sensors_camera:

.. currentmodule:: isaaclab

Camera
======

A :class:`~sensors.Camera` defines what to capture: camera pose, projection, resolution, sampling
period, and output data types. A renderer defines how those images are produced. Keeping these
responsibilities separate lets one camera configuration work with different physics and rendering
backends.

Camera data is expensive compared with low-dimensional state. Isaac Lab therefore batches the
camera copies from cloned environments into tiled render passes and exposes the de-tiled result as
one device-resident buffer per requested output.

For silent black or corrupted RTX camera images when scaling the workload, see
:ref:`known-issues-rtx-buffer-limit`.

Rendering model
---------------

:attr:`~sensors.CameraCfg.renderer_cfg` selects the renderer. A plain
:class:`~isaaclab.renderers.RendererCfg` requests the runtime default. Use a concrete configuration
when the renderer must be fixed:

.. list-table::
   :header-rows: 1
   :widths: 34 23 43

   * - Renderer configuration
     - Requires Isaac Sim
     - Characteristics
   * - :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`
     - Yes
     - Replicator and RTX rendering through Isaac Sim
   * - :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`
     - No
     - Kit-less RTX rendering through ``isaaclab_ov``
   * - :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`
     - No
     - Kit-less Warp rasterization through Newton

For an environment that exposes renderer presets, select the renderer at launch instead of editing
the scene configuration. Pass ``physics=`` and ``renderer=`` together, since the renderer must match
your setup path. The command below is for the kit-less Newton path; for Isaac Sim, use
``physics=isaacsim_physx renderer=isaacsim_rtx`` with ``--extra isaacsim``. See
:doc:`/source/how-to/configure_rendering` for each pairing:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl \
             --task Isaac-Cartpole-Camera-Direct \
             physics=newton_mjwarp renderer=newton_renderer presets=rgb

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run isaaclab train --rl_library rsl_rl ^
             --task Isaac-Cartpole-Camera-Direct ^
             physics=newton_mjwarp renderer=newton_renderer presets=rgb

Images are not written to disk by default. Set ``env.write_image_to_file=true`` to write
``cartpole_<data_type>.png`` in the current working directory.

See :doc:`/source/concepts/backends_and_presets` for preset discovery and
:ref:`renderer-visual-comparison` for a same-scene comparison of the renderer outputs.

With OVRTX, separate camera sensors share the scene and renderer when their renderer
configurations match, while each sensor owns a separate tiled render product. For example,
``env_0/head_camera`` and ``env_1/head_camera`` form one product, while
``env_0/wrist_camera`` and ``env_1/wrist_camera`` form another. The sensors can use different
resolutions and output types, and their poses update independently. Define all camera prims
before initializing the simulation so they are included in the scene exported to OVRTX.

.. _camera-configuration:

Configure a camera
------------------

A camera can spawn a pinhole or fisheye camera prim, or bind to a camera already on the stage.
``offset`` uses the convention declared on :class:`~sensors.CameraCfg.OffsetCfg`:

* ``world``: forward ``+X``, up ``+Z``.
* ``ros``: forward ``+Z``, up ``-Y``.
* ``opengl``: forward ``-Z``, up ``+Y``.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.sensors import CameraCfg
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   front_camera = CameraCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
       update_period=0.05,
       height=240,
       width=320,
       data_types=["rgb", "depth", "normals"],
       spawn=sim_utils.PinholeCameraCfg(
           focal_length=24.0,
           horizontal_aperture=20.955,
           clipping_range=(0.1, 20.0),
       ),
       offset=CameraCfg.OffsetCfg(
           pos=(0.45, 0.0, 0.1),
           rot=(0.0, 0.0, 0.0, 1.0),
           convention="world",
       ),
       renderer_cfg=NewtonWarpRendererCfg(),
   )

A renderer instance is reused only when cameras use equal renderer configurations of the same
concrete configuration type. Different renderer settings create distinct instances. Camera
configurations, including output types and backgrounds, remain per sensor.

Read camera data
----------------

``camera.get_world_poses(convention="opengl")`` reads the current world poses without capturing images
or changing cached measurement poses. This includes motion of the camera's parent body even when
``update_latest_camera_pose=False``. It accepts the same orientation conventions as
``set_world_poses`` and returns device tensors for positions and quaternions.

:attr:`~sensors.CameraData.output` maps each requested name to a
:class:`~isaaclab.utils.warp.ProxyArray`. For ``N`` camera views, height ``H``, width ``W``, and
``C`` channels, each output has shape ``(N, H, W, C)``. Use ``torch`` for a cached zero-copy Torch
view or ``warp`` for the underlying Warp array:

.. code-block:: python

   camera_data = scene["front_camera"].data
   rgb = camera_data.output["rgb"].torch
   depth = camera_data.output["depth"].torch
   intrinsics = camera_data.intrinsic_matrices.torch

Camera pose and intrinsic buffers are also ``ProxyArray`` objects. ``pos_w`` has shape ``(N, 3)``,
``intrinsic_matrices`` has shape ``(N, 3, 3)``, and camera quaternions have shape ``(N, 4)`` in
``(x, y, z, w)`` order. Set ``update_latest_camera_pose=True`` only when current pose data is needed;
updating it adds frame-query overhead.

.. _camera-output-types:

Output types
------------

For raw outputs, the active renderer validates the requested names and allocates the channel count
and data type declared by its :class:`~isaaclab.renderers.RenderBufferSpec`. A camera pipeline may
produce additional output names; its processors allocate those buffers.

.. list-table:: Common output contracts
   :header-rows: 1
   :widths: 34 22 44

   * - Name
     - Channels and type
     - Meaning
   * - ``rgb`` / ``rgba``
     - 3 / 4, ``uint8``
     - Low-dynamic-range color
   * - ``rgb_hdr``
     - 3, ``float32``
     - High-dynamic-range RGB using the active camera settings
   * - ``albedo``
     - 4, ``uint8``
     - Material base color
   * - ``depth`` / ``distance_to_image_plane``
     - 1, ``float32``
     - Distance [m] along the camera optical axis
   * - ``distance_to_camera``
     - 1, ``float32``
     - Euclidean distance [m] from the optical center
   * - ``normals``
     - 3, ``float32``
     - Local surface normal ``(x, y, z)``
   * - ``motion_vectors``
     - 2, ``float32``
     - Image-space motion; positive ``x`` is left and positive ``y`` is up
   * - ``semantic_segmentation``
     - 4 ``uint8`` or 1 ``int32``
     - Semantic color or ID per pixel
   * - ``instance_segmentation``
     - 4 ``uint8`` or 1 ``int32``
     - Semantically labeled instance color or ID per pixel
   * - ``instance_id_segmentation_fast``
     - 4 ``uint8`` or 1 ``int32``
     - USD-prim instance color or ID per pixel

``depth`` is an alias of ``distance_to_image_plane``. Colorized segmentation uses RGBA ``uint8``;
non-colorized segmentation uses one ``int32`` ID channel. Label and prim-path mappings are stored in
``camera_data.info[output_name]``.

``rgb_hdr`` uses the camera's active exposure settings. PPISP requires scene-linear HDR;
set ``CameraCfg.exposure=CameraExposureCfg()`` explicitly to disable RTX automatic exposure and
use neutral exposure. Leaving ``exposure=None`` preserves authored camera settings. Selecting an
output never changes exposure. PPISP requests ``rgb_hdr`` privately; include it in ``data_types``
only to publish the raw input as well.

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/camera-renderer-isaac-rtx.webp
   :align: center
   :figwidth: 100%
   :alt: RGB camera output

   Isaac RTX RGB output. The animation shows the six material spheres falling onto the table.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-depth.png
   :align: center
   :figwidth: 100%
   :alt: Depth camera output

   Isaac RTX depth output. Display colors encode optical-axis distance; the sensor returns metric
   values [m].

.. _camera-supported-annotators:

Renderer support
~~~~~~~~~~~~~~~~

The common API does not imply that every renderer produces every output. For same-scene examples
across these backends, see the :ref:`renderer visual comparison <renderer-visual-comparison>`. The
current support matrix is:

.. list-table::
   :header-rows: 1
   :widths: 38 20 20 22

   * - Output
     - Isaac RTX
     - OVRTX
     - Newton Warp
   * - ``rgb``, ``rgba``, ``rgb_hdr``
     - Yes
     - Yes
     - Yes
   * - ``depth`` and both distance outputs
     - Yes
     - Yes
     - Yes
   * - ``normals``
     - Yes
     - Yes
     - Yes
   * - ``albedo``
     - Isaac Sim 6.0+
     - Yes
     - Yes
   * - ``motion_vectors``
     - Yes
     - Yes
     - No
   * - semantic and instance segmentation
     - Yes
     - Yes
     - Yes
   * - ``instance_id_segmentation_fast``
     - Yes
     - No
     - No
   * - ``simple_shading_*`` modes
     - Isaac Sim 6.0+
     - Yes
     - No

.. note::

   Explicit exposure settings affect every output from the camera prim, including other sensors
   sharing it. Use separate camera prims when different exposures are required. Newton's HDR
   output is already scene-linear and does not consume RTX exposure attributes.

Querying an unsupported output fails during camera initialization. Renderer configuration controls
semantic filters, segmentation colorization, and depth clipping where those options are
backend-specific.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-normals.png
   :align: center
   :figwidth: 100%
   :alt: Camera surface-normal output

   Isaac RTX normals output. Red, green, and blue encode the surface normal X, Y, and Z components.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-semantic-segmentation.png
   :align: center
   :figwidth: 100%
   :alt: Semantic segmentation output

   Isaac RTX semantic segmentation. One color represents each class: the six spheres share one
   class, while the table and backdrop use separate classes.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-instance-segmentation.png
   :align: center
   :figwidth: 100%
   :alt: Instance segmentation output

   Isaac RTX instance segmentation. Each sphere receives its own color, distinguishing objects
   that share the same semantic class.

Background color
----------------

When :attr:`~sensors.CameraCfg.background_color` is ``None``, each renderer uses its default
background. Set a normalized RGB tuple to use a solid color for pixels that miss all geometry:

.. code-block:: python

   from isaaclab.sensors.camera import CameraExposureCfg
   from isaaclab.utils import replace

   mask_camera = replace(front_camera, background_color=(0.0, 0.0, 0.0))

The setting is per camera. Cameras with renderer-default and solid backgrounds can coexist in one
scene.

Post-render image processing
----------------------------

:attr:`~sensors.CameraCfg.pipeline` composes dictionary processors after rendering.
:class:`~isaaclab.utils.tensor_processors.DictTensorProcessor` takes a dictionary of named tensors
and returns a dictionary of named tensors. It lives beside
:class:`~isaaclab.utils.tensor_processors.TensorProcessor`, whose call takes and returns one tensor.
:class:`~isaaclab.utils.tensor_processors.DictTensorProcessorCfg` declares the actual input and output
keys needed for renderer setup. There are no separate operation roles or prepared-operation objects.
The shared dictionary interface also accepts nonvisual sensor data; each processor defines its
layout and value semantics.

The camera resolves external inputs before renderer initialization, including inputs needed only by the
pipeline. It publishes the names in ``data_types`` after processing. For example, this camera requests
private ``rgb_hdr`` from its renderer and publishes PPISP's processed ``rgb``:

.. code-block:: python

   from isaaclab.sensors.camera import CameraExposureCfg
   from isaaclab.utils import replace
   from isaaclab_ppisp import PpispCfg, PpispProcessorCfg

   processed_camera = replace(
       front_camera,
       data_types=["rgb"],
       exposure=CameraExposureCfg(),
       pipeline=[PpispProcessorCfg(params={"cfg": PpispCfg(inputs={"exposureOffset": 1.5})})],
   )

PPISP (Physically Plausible Image Signal Processing) applies responsivity, exposure, vignetting, color
correction, and a camera response function. It consumes float32 scene-linear radiance and produces
uint8 color. Set ``outputs=["rgba"]`` and ``params={"cfg": ppisp_cfg, "channels": 4}`` on the
step for RGBA output. A pipeline-produced RGBA buffer also supplies a zero-copy RGB view when ``rgb``
is requested. A pipeline producing only RGB cannot satisfy an explicit RGBA request.

For custom dictionary processing, subclass ``DictTensorProcessor``. Its constructor receives
``(cfg, data)`` with actual ``ProxyArray`` tensors. Allocate persistent output tensors in
``self.outputs`` without reading input values or advancing state. Implement ``__call__(data)``
to fill and return those named outputs, and override ``reset`` or ``close`` only when needed.
The pipeline passes the current dictionary to each processor and merges its returned outputs;
untouched keys pass through. Shape, dtype and device come from the arrays directly.

PPISP is a regular ``PpispProcessor`` subclass. It can also be constructed and called directly:

.. code-block:: python

   from isaaclab.utils import instantiate

   cfg = PpispProcessorCfg()
   processor = instantiate(cfg, {"rgb_hdr": hdr_tensor})
   color = processor({"rgb_hdr": hdr_tensor})["rgb"]
   processor.close()

Here ``hdr_tensor`` is a scene-linear float32 ``ProxyArray`` with shape ``(N, H, W, 3)``.
The returned color is uint8; copy it before another call if its values must be retained.

Published camera outputs must keep the camera's batch size, height, width, and NHWC layout so pose
and calibration metadata remain valid. Intermediate buffers may change shape. Use ``processed_sensor``
below when the final observation should have a different image shape or layout.

``camera.render_outputs`` exposes borrowed raw renderer buffers, including private inputs. Their
``ProxyArray`` properties provide shape, dtype, and device. Reading raw outputs updates the camera
when due. Neither an operation nor a caller may mutate borrowed inputs. Public processed
buffers and raw buffers retain their storage until camera reinitialization, but later captures replace
their contents; clone values that must survive another capture. Each camera owns its pipe and operation
workspace. Operations may share explicitly supplied immutable model resources.
Configuration copying must preserve the identity of borrowed resources: PPISP provides
``PpispControllerWeights`` for device weights. Custom processors can use a resource wrapper with
identity-preserving ``__deepcopy__``; raw Warp allocations are not configuration values.

Pipelines run after scalar and batched captures. When an asynchronous renderer republishes a capture,
the camera retains its processed result and matching capture metadata. Camera reset forwards only the
selected environment indices to operation reset callbacks; teardown closes the operations. For an
observation-only pipeline, declare raw dependencies with ``camera.request_render_inputs(names)`` before
simulation initialization. Additional requests after initialization fail because renderer setup is
already complete. Observation consumers use their own pipe and snapshot results needed across captures.

Processing sensor observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~isaaclab.envs.mdp.processed_sensor` for observation-only processing, including operations
that combine several camera outputs or change image dimensions. The term currently binds sensors
through their ``request_render_inputs()`` and ``render_outputs`` interface:

.. code-block:: python

   from isaaclab.envs import mdp
   from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
   from isaaclab_ppisp import PpispProcessorCfg

   front_camera.exposure = CameraExposureCfg()
   policy_image = ObservationTermCfg(
       func=mdp.processed_sensor,
       params={
           "sensor_cfg": SceneEntityCfg("front_camera"),
           "pipeline": [PpispProcessorCfg()],
           "output_name": "rgb",
       },
   )

Manager-based environments collect the term's private renderer inputs before simulation reset. A
standalone observation manager must call ``ObservationManager.prepare_scene(cfg, env)`` before reset.
The term processes raw renderer outputs even when the camera has its own pipeline. To read the
camera's published processed output, use an existing image term such as ``mdp.image_rgb`` instead.
The observation pipe executes on every term evaluation and returns a snapshot; its state belongs to
that term rather than the camera's capture cadence.

For a single incoming tensor, :class:`~isaaclab.utils.tensor_processors.DictTensorProcessorAdapterCfg` uses the
same operations while preserving preceding observation transformations. This includes nonvisual
observation tensors from sensors without the render-input interface. This form must preserve shape;
use ``processed_sensor`` for multiple render inputs or changes to render-output shapes, or construct
a ``DictTensorProcessor`` directly for other named buffers. The first incoming tensor prepares the
operations; later calls reuse their output storage. Its values must meet the operation's documented
input requirements, including any transformations from preceding tensor processors:

.. code-block:: python

   from isaaclab.utils import tensor_processors

   # This camera publishes radiance for the ordinary image term to read.
   raw_camera = replace(front_camera, data_types=["rgb_hdr"], exposure=CameraExposureCfg(), pipeline=[])
   policy_image = ObservationTermCfg(
       func=mdp.image_rgb,
       params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb_hdr", "normalize": False},
       tensor_processors=[
           tensor_processors.DictTensorProcessorAdapterCfg(
               pipeline=[PpispProcessorCfg()],
               input_name="rgb_hdr",
               output_name="rgb",
           )
       ],
   )

Migrating renderer-owned ISP configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CameraCfg.isp_cfg`` no longer executes image processing. A non-None value raises a migration error;
replace it with ``CameraCfg.pipeline`` as above. ``CameraISPMode`` remains importable for legacy
configuration values, but new code should resolve USD attributes explicitly before constructing the
operation:

.. code-block:: python

   from isaaclab_ppisp import PpispDiscoveryMode, PpispProcessorCfg, resolve_and_normalize

   ppisp_cfg = resolve_and_normalize(PpispDiscoveryMode.AUTO_CAMERA, stage, "/World/Camera")
   discovered_camera = replace(
       front_camera,
       data_types=["rgb"],
       exposure=CameraExposureCfg(),
       pipeline=[PpispProcessorCfg(params={"cfg": ppisp_cfg})] if ppisp_cfg is not None else [],
   )

``AUTO_CAMERA`` checks the supplied camera prim; ``AUTO_ANY`` can fall back to the first PPISP camera
on the supplied stage. Alternatively, pass an already resolved camera prim to
``ppisp_cfg_from_usd_camera(camera_prim)``. Discovery has no implicit simulation or stage lookup inside
the operation. Static coefficients apply to the whole camera batch; controller weights may predict
per-view exposure and color parameters.

For either processing attachment, configure the source camera with
``exposure=CameraExposureCfg()`` before simulation initialization. PPISP's own exposure is set
through ``PpispCfg.inputs["exposureOffset"]`` (stops). Processing does not discover or change
renderer exposure settings.

Run the ``ppisp-camera`` example for a complete PPISP workflow. It supports
``--renderer newton_renderer`` (Newton Warp, the default) and ``--renderer isaac_rtx`` (Isaac RTX).
The example runs 0 warmup steps for Newton Warp and 32 for Isaac RTX before saving images; override
this with ``--warmup_steps``.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run --extra isaacsim isaaclab example ppisp-camera \
             --renderer newton_renderer --max_steps 60

         uv run --extra isaacsim isaaclab example ppisp-camera \
             --input_scene /path/to/scene.usd --renderer isaac_rtx \
             --camera_prim_path /World/Cameras/front_cam --max_steps 60

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run --extra isaacsim isaaclab example ppisp-camera ^
             --renderer newton_renderer --max_steps 60

         uv run --extra isaacsim isaaclab example ppisp-camera ^
             --input_scene C:/path/to/scene.usd --renderer isaac_rtx ^
             --camera_prim_path /World/Cameras/front_cam --max_steps 60

``--input_scene`` accepts a local path or URL to a USD or USDZ scene containing the Gaussian PPISP
setup. If omitted, the example uses the ``Samples/Scene_ParticleField/valiant_auto.usdz``
sample from ``ISAAC_NUCLEUS_DIR``.

``--camera_prim_path`` is an optional absolute USD prim path in the input scene. The prim must exist,
have type ``Camera``, and carry ``ppisp:*`` attributes. If omitted, the example selects the first
camera with PPISP attributes. For both renderers, the input scene must have a ``defaultPrim`` and
the camera must be a descendant of it (for example, ``/World`` for the camera path above).

Images are saved to ``output/ppisp_camera`` in the current working directory; use ``--output_dir``
to change it. ``--max_steps`` sets the number of steps after warmup (default: 120), and
``--save_interval`` sets the image-saving interval in steps (default: 20).
For all options, run ``uv run --extra isaacsim isaaclab example ppisp-camera --help``.

Performance and validation
--------------------------

Image memory and rendering cost scale with the number of environments, resolution, channel count,
and requested outputs. Start camera-based tasks with a small environment count, verify shapes and
renderer support, and then scale while monitoring GPU memory. Avoid requesting buffers that the task
does not consume.

Tiled rendering batches the cloned views into shared render passes, but it does not remove the memory
cost of the de-tiled outputs or downstream vision models. The camera follows the shared sensor
``update_period`` contract; choose a period that matches the observation cadence instead of rendering
at every physics step by default.

A runnable camera example is available as ``camera``:

.. code-block:: bash

   uv run --extra isaacsim isaaclab example camera

For saving output to disk, see :doc:`/source/how-to/save_camera_output`. For renderer selection
and customization, see :doc:`/source/how-to/configure_rendering`.

The observation processor API is ``isaaclab.utils.tensor_processors``. Custom stateful processors
inherit ``TensorProcessor`` and use ``TensorProcessorCfg``; attach configurations through
``ObservationTermCfg.tensor_processors``. These replace ``ModifierBase``, ``ModifierCfg``, and
``ObservationTermCfg.modifiers`` respectively. ``DictTensorProcessorAdapterCfg`` adapts a dictionary
processor pipeline to that single-tensor interface.
