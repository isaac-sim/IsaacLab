.. _overview_sensors_camera:

.. currentmodule:: isaaclab

Camera
======

Camera sensors in Isaac Lab are renderer-backed sensors: each :class:`~sensors.Camera` instance
is coupled to a **renderer** that produces the image data. If multiple cameras use the same renderer
type, only one renderer is instantiated and shared between them. The renderer and camera are intentionally
isolated from each other — the camera defines *what* to capture (pose, resolution, field of view,
data types), while the renderer defines *how* to render it (RTX ray-tracing, Newton Warp rasterizer,
etc.). This separation allows the same camera configuration to run across different physics and
rendering backends without code changes.

For an overview of the available renderer backends and how to choose between them, see
:ref:`overview_renderers`.

Rendered images are unique among supported sensor data types due to their large bandwidth requirements.
A single 800 × 600 image with 32-bit color clocks in at just under 2 MB. At 60 fps across thousands
of parallel environments, this grows quickly. Isaac Lab's tiled rendering API specifically addresses
these scaling challenges by batching all cameras into a single render pass.


Renderer Backends
-----------------

The renderer used by a camera is configured via the ``renderer_cfg`` field on
:class:`~sensors.CameraCfg`. The default is :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`
(NVIDIA RTX, requires Isaac Sim).

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - ``renderer_cfg``
     - Requires Isaac Sim?
     - Supported data types
   * - ``IsaacRtxRendererCfg`` *(default)*
     - Yes
     - rgb, rgba, depth, normals, motion vectors, semantic/instance segmentation, and all other annotators
   * - ``NewtonWarpRendererCfg``
     - No (kit-less)
     - ``rgb``, ``depth`` only
   * - ``OVRTXRendererCfg``
     - No (+ ``isaaclab_ov``)
     - ``rgb``, ``depth`` only

.. note::

   The Newton Warp renderer currently supports only **``rgb``** and **``depth``** data types.
   Annotators such as segmentation, normals, and motion vectors are Isaac RTX-specific features and
   require :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`.


Tiled Rendering
~~~~~~~~~~~~~~~

.. note::

    This feature is available from Isaac Sim version 4.2.0 onwards (for the RTX renderer).
    The Newton Warp renderer supports tiled rendering in kit-less mode.

    Tiled rendering in combination with image processing networks require heavy memory resources,
    especially at larger resolutions. We recommend running 512 cameras on RTX 4090 GPUs or similar
    when using the RTX renderer.

The Tiled Rendering API provides a vectorized interface for collecting image data from all environment
clones in a single batched render pass. Instead of one render call per camera, all copies of a camera
are composited into a single large tiled image, dramatically reducing host-device transfer overhead.

Isaac Lab provides tiled rendering through :class:`~sensors.Camera`, configured via
:class:`~sensors.CameraCfg`. The ``renderer_cfg`` field selects the rendering backend.


CameraCfg with renderer_cfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The renderer is specified via ``renderer_cfg`` on :class:`~sensors.CameraCfg`. The camera and
renderer configurations are fully decoupled: you can swap renderers without changing any other camera
parameters.

**Default (RTX, requires Isaac Sim):**

.. code-block:: python

    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    # IsaacRtxRendererCfg is the default, no explicit import needed

    tiled_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
        # renderer_cfg defaults to IsaacRtxRendererCfg()
    )

**Newton Warp renderer (kit-less, no Isaac Sim required):**

.. code-block:: python

    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    import isaaclab.sim as sim_utils

    tiled_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb", "depth"],  # only rgb and depth supported with Newton renderer
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
        renderer_cfg=NewtonWarpRendererCfg(),
    )

**Multi-backend preset (switches renderer alongside physics backend):**

For environments that need to support both backends, use
:class:`~isaaclab_tasks.utils.presets.MultiBackendRendererCfg` together with the
:ref:`PresetCfg pattern <migrating-to-isaaclab-3-0>`:

.. code-block:: python

    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.utils.presets import MultiBackendRendererCfg
    import isaaclab.sim as sim_utils

    tiled_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
        renderer_cfg=MultiBackendRendererCfg(),  # selects RTX or Newton Warp via presets= CLI arg
    )

The active preset is selected at launch via ``physics=``, ``renderer=``, or ``presets=`` CLI arguments:

.. code-block:: bash

   # Use Newton Warp renderer
   python train.py task=Isaac-Cartpole-Camera-Direct renderer=newton_renderer

   # Use OVRTX renderer
   python train.py task=Isaac-Cartpole-Camera-Direct renderer=ovrtx_renderer

   # Use default (Isaac RTX)
   python train.py task=Isaac-Cartpole-Camera-Direct


Accessing camera data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    tiled_camera = Camera(cfg.tiled_camera)
    data = tiled_camera.data.output["rgb"]  # shape: (num_cameras, H, W, 3), torch.uint8

The returned data has shape ``(num_cameras, height, width, num_channels)``, ready to use directly
as an observation in RL training.

When using the RTX renderer, add ``--enable_cameras`` when launching:

.. code-block:: shell

    ./isaaclab.sh train --rl_library rl_games \
        --task=Isaac-Cartpole-Camera-Direct --enable_cameras


Annotators (RTX only)
~~~~~~~~~~~~~~~~~~~~~

.. note::

   Annotators are a feature of the **Isaac RTX renderer** (``IsaacRtxRendererCfg``).
   They are **not** available with the Newton Warp renderer or ovrtx, which
   support only ``rgb`` and ``depth``.

:class:`~sensors.Camera` exposes the following annotator
data types when using the RTX renderer:

* ``"rgb"``: A 3-channel rendered color image.
* ``"rgba"``: A 4-channel rendered color image with alpha channel.
* ``"distance_to_camera"``: Distance to the camera optical center per pixel.
* ``"distance_to_image_plane"``: Distance along the camera's Z-axis per pixel.
* ``"depth"``: Alias for ``"distance_to_image_plane"``.
* ``"normals"``: Local surface normal vectors at each pixel.
* ``"motion_vectors"``: Per-pixel motion vectors in image space.
* ``"semantic_segmentation"``: Semantic segmentation labels.
* ``"instance_segmentation_fast"``: Instance segmentation data.
* ``"instance_id_segmentation_fast"``: Instance ID segmentation data.

RGB and RGBA
~~~~~~~~~~~~

.. figure:: ../../../_static/overview/sensors/camera_rgb.jpg
    :align: center
    :figwidth: 100%
    :alt: A scene captured in RGB

``rgb`` returns a 3-channel RGB image of type ``torch.uint8``, shape ``(B, H, W, 3)``.

``rgba`` returns a 4-channel RGBA image of type ``torch.uint8``, shape ``(B, H, W, 4)``.

To convert to ``torch.float32``, divide by 255.0.

``rgb_hdr`` returns a 3-channel scene-linear HDR image of type ``torch.float32``, shape ``(B, H, W, 3)``.

Post-render Camera ISP
~~~~~~~~~~~~~~~~~~~~~~

A camera Image Signal Processing (ISP) pipeline models the chain that maps
the scene-linear radiance captured by a sensor to the LDR pixel values a
downstream consumer sees.
The camera ISP pipeline is usually part of the renderer.
In Isaac Lab we expose a post-render camera ISP pipeline which is applied on top of the renderer's HDR scene-linear AOV.
This makes it possible to implement additional post-render processing not currently supported by the renderer backends.
The pass is configured via :attr:`~sensors.CameraCfg.isp_cfg`
on every camera and runs once per render tick.

PPISP
^^^^^

The shipped ISP implementation is **PPISP** (Physically Plausible Image
Signal Processing), an NVIDIA Spatial Intelligence Lab pipeline designed
to bring synthetic imagery — most notably 3D Gaussian splat reconstructions
— closer to real-camera output without re-training the upstream model. See
the project page: https://research.nvidia.com/labs/sil/projects/ppisp .

PPISP is typically authored alongside a `ParticleField3DGaussianSplat
<https://openusd.org/release/user_guides/schemas/usdVol/ParticleField3DGaussianSplat.html>`__
USD asset: its camera prim authors ``ppisp:*`` attributes calibrated against
the real capture rig that produced the splats. Configuring the camera with
the matching PPISP coefficients makes the rendered tile match the calibration
target.

With static coefficients, the image pass applies, in order: responsivity →
exposure → vignetting → color homography → camera response function → uint8
clamp. It runs as a single Warp image kernel.

When the camera also authors ``ppisp:controllerWeights``, Isaac Lab runs the
exported PPISP controller before the image pass. The controller reads the
current HDR image, prior exposure, and controller responsivity, then predicts
``exposureOffset`` and the four color-latent pairs for each camera view. The
image pass then uses those predicted exposure and color-latent values while
static PPISP inputs still provide responsivity, vignetting, and CRF.

Configuration
^^^^^^^^^^^^^

:attr:`~sensors.CameraCfg.isp_cfg` accepts three forms:

* ``None`` (default) — ISP disabled.
* :class:`~isaaclab_ppisp.PpispCfg` — explicit PPISP coefficients
  (:attr:`~isaaclab_ppisp.PpispCfg.inputs`) and, optionally, controller
  weights. Use :attr:`~isaaclab_ppisp.PpispCfg.camera_prim_path` to import
  static coefficients and camera-authored controller weights from a USD camera
  already on the stage.
* :class:`~sensors.CameraISPMode` — auto-discover ISP camera attributes on the
  stage (see below).

The cfg applies once per Camera sensor batch. Static PPISP inputs are scalar
coefficients shared by every cloned view in a tiled batch. When controller
weights are present, the controller predicts per-view exposure and color-latent
values from each HDR image, while responsivity, vignetting, and CRF remain
shared by the batch.

.. code-block:: python

   from isaaclab.sensors.camera import CameraCfg, CameraISPMode
   from isaaclab_ppisp import PpispCfg

   # default — ISP disabled
   cfg = CameraCfg(...)

   # explicit coefficients
   cfg = CameraCfg(..., isp_cfg=PpispCfg(inputs={"exposureOffset": 1.5}))

   # import coefficients from a USD camera path
   cfg = CameraCfg(..., isp_cfg=PpispCfg(camera_prim_path="/World/Camera_ppisp"))

   # auto-discover from the stage
   cfg = CameraCfg(..., isp_cfg=CameraISPMode.AUTO_ANY)

Auto-discovery
^^^^^^^^^^^^^^

Auto-discovery is opt-in via :class:`~sensors.CameraISPMode`. Discovery runs
once at camera construction using the first matched camera prim in the Camera
sensor batch:

1. Check the first matched camera prim for ``ppisp:*`` attributes.
2. ``AUTO_ANY`` only, or when no camera path is available: fall back to the
   first camera anywhere on the stage with ``ppisp:*`` attributes.
3. Otherwise the ISP stays disabled for the whole Camera sensor batch.

In practice this means: if the stage carries a ``ParticleField3DGaussianSplat``
together with a camera that authors ``ppisp:*`` attributes, the Camera sensor
picks up the matching ISP automatically, including controller weights when
authored, and no Python-side coefficient authoring is required.

``AUTO_CAMERA`` runs only the camera-local discovery steps — useful when the
stage carries multiple PPISP cameras and you want the Camera sensor batch to use
the attributes authored on its first matched camera prim.

Renderer support
^^^^^^^^^^^^^^^^

All three shipped backends advertise the HDR AOV
(:attr:`~renderers.RenderBufferKind.RGB_HDR`) and compose the ISP pipeline
internally: the Isaac RTX renderer sources HDR from the Replicator
``HdrColor`` annotator, the OVRTX renderer from its HDR render var, the
Newton Warp renderer from its native scene-linear color buffer. Each
backend allocates its own HDR scratch buffer when the user did not request
``"rgb_hdr"`` in :attr:`~sensors.CameraCfg.data_types`, and dispatches the
PPISP pipeline into ``rgb`` / ``rgba`` after every render tick. For controller
configs, this is the controller pass followed by the PPISP image pass.

Usage example
^^^^^^^^^^^^^

For a runnable usage example, see ``scripts/demos/sensors/ppisp_camera.py``.
It loads a PPISP-authored USD or USDZ Gaussian scene, creates baseline and
PPISP camera sensors for the selected camera, and saves baseline, PPISP, and
absolute-difference images.

.. code-block:: bash

   ./isaaclab.sh -p scripts/demos/sensors/ppisp_camera.py \
       --renderer newton --max_steps 60

Use ``--renderer isaac_rtx`` to run the same workflow with Isaac RTX. Pass
``--input_scene`` for a custom scene and ``--camera_prim_path`` if the stage
contains multiple cameras with PPISP attributes. If a config or command selects
a visualizer, force-disable all visualizers with ``--visualizer none`` or
``--viz none``. Images are written to
``scripts/demos/sensors/output/ppisp_camera`` unless ``--output_dir`` is set.

Known limitations
^^^^^^^^^^^^^^^^^

* The ISP writes back into the ``rgb`` / ``rgba`` buffers. If neither is
  requested, configuring ``isp_cfg`` raises at camera init.
* Static PPISP inputs and controller weights are fixed for the lifetime of the
  camera. Animated USD camera attributes are collapsed to their first authored
  time sample.
* Static coefficients are global per Camera sensor batch — no per-pixel or
  per-region authoring beyond the radial vignetting term.
* PPISP is the only ISP implementation today. Other ISP families would
  need a new config type and discoverer entry.
* On the Isaac RTX and OVRTX backends, enabling ``isp_cfg`` forces RTX-side
  tonemapping off (``/rtx/rtpt/gaussian/skipTonemapping/enabled=False``)
  and authors a neutral ``OmniRtxCameraExposureAPI_1`` schema on each
  camera prim so the post-render ISP is the only path that processes
  color. Mixing this with RTX-side exposure authoring is not supported.
* Auto-discovery resolves at camera construction; later authoring of
  ``ppisp:*`` camera attributes on the stage is not picked up.

Depth and Distances
~~~~~~~~~~~~~~~~~~~

.. figure:: ../../../_static/overview/sensors/camera_depth.jpg
    :align: center
    :figwidth: 100%
    :alt: A scene captured as depth

``distance_to_camera`` returns a single-channel depth image with distance to the camera optical
center, shape ``(B, H, W, 1)``, type ``torch.float32``.

``distance_to_image_plane`` returns distances of 3D points from the camera plane along the Z-axis,
shape ``(B, H, W, 1)``, type ``torch.float32``.

``depth`` is an alias for ``distance_to_image_plane``.

Normals
~~~~~~~

.. figure:: ../../../_static/overview/sensors/camera_normals.jpg
    :align: center
    :figwidth: 100%
    :alt: A scene captured with surface normals

``normals`` returns local surface normal vectors at each pixel, shape ``(B, H, W, 3)`` containing
``(x, y, z)``, type ``torch.float32``.

Motion Vectors
~~~~~~~~~~~~~~

``motion_vectors`` returns per-pixel motion vectors in image space between frames.
Shape ``(B, H, W, 2)``: ``x`` is horizontal motion (positive = left), ``y`` is vertical motion
(positive = up). Type ``torch.float32``.

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../../_static/overview/sensors/camera_semantic.jpg
    :align: center
    :figwidth: 100%
    :alt: A scene with semantic segmentation

``semantic_segmentation`` outputs per-pixel semantic labels for entities with semantic annotations.
An ``info`` dictionary is available via ``tiled_camera.data.info['semantic_segmentation']``.

- If ``colorize_semantic_segmentation=True``: 4-channel RGBA image, shape ``(B, H, W, 4)``,
  type ``torch.uint8``. The ``idToLabels`` dict maps color to semantic label.
- If ``colorize_semantic_segmentation=False``: shape ``(B, H, W, 1)``, type ``torch.int32``,
  containing semantic IDs. The ``idToLabels`` dict maps ID to label.

Instance ID Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../../_static/overview/sensors/camera_instanceID.jpg
    :align: center
    :figwidth: 100%
    :alt: A scene with instance ID segmentation

``instance_id_segmentation_fast`` outputs per-pixel instance IDs, unique per USD prim path.
An ``info`` dictionary is available via ``tiled_camera.data.info['instance_id_segmentation_fast']``.

- If ``colorize_instance_id_segmentation=True``: shape ``(B, H, W, 4)``, type ``torch.uint8``.
  The ``idToLabels`` dict maps color to USD prim path.
- If ``colorize_instance_id_segmentation=False``: shape ``(B, H, W, 1)``, type ``torch.int32``.
  The ``idToLabels`` dict maps instance ID to USD prim path.

Instance Segmentation
~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../../_static/overview/sensors/camera_instance.jpg
    :align: center
    :figwidth: 100%
    :alt: A scene with instance segmentation

``instance_segmentation_fast`` outputs instance segmentation, traversing down the prim hierarchy
to the lowest level with semantic labels (unlike ``instance_id_segmentation_fast``, which always
goes to the leaf prim).
An ``info`` dictionary is available via ``tiled_camera.data.info['instance_segmentation_fast']``.

- If ``colorize_instance_segmentation=True``: shape ``(B, H, W, 4)``, type ``torch.uint8``.
- If ``colorize_instance_segmentation=False``: shape ``(B, H, W, 1)``, type ``torch.int32``.

The ``idToLabels`` dict maps color to USD prim path. The ``idToSemantics`` dict maps color to
semantic label.
