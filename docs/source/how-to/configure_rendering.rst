Configuring RTX Rendering Settings
====================================

.. note::

   This guide covers the **RTX renderer** settings, which are used when running Isaac Lab with
   Isaac Sim. The RTX renderer is based on NVIDIA's Omniverse RTX rendering pipeline and is
   available for all camera sensors in the PhysX backend.

   For the **Newton renderer** (used with the Newton backend or in kit-less mode), see
   :ref:`overview_renderers` for the pluggable renderer architecture and available backends.

Isaac Lab's RTX renderer offers 3 preset rendering modes: performance, balanced, and quality.
You can select a mode via a command line argument or from within a script, and customize settings as needed.
Adjust and fine-tune rendering to achieve the ideal balance for your workflow.

Selecting a Rendering Mode
--------------------------

Note, at present, rendering-mode profiles are supported for Kit Visualizers and RTX based Renderers

You can pick ``performance``, ``balanced``, or ``quality`` (or a custom name you added under
:attr:`~sim.SimulationCfg.rendering_mode_cfgs`) in a few places: from the **command line**, through **visualizer configs**, or
through **renderer configs**.

1. **Command Line.** ``--rendering_mode`` sets the mode at launch and overrides visualizer and RTX camera renderer
   config settings when you pass it.

   .. code-block:: bash

      ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py --rendering_mode {performance/balanced/quality}

2. **Visualizer Configs.** From a Visualizer Config, set the ``rendering_mode`` field to a profile name defined by the
   the RenderingMode Configs. There are 3 presets: ``performance``, ``balanced``, and ``quality``. Users can also define
   custom profiles by adding a named entry to :attr:`~sim.SimulationCfg.rendering_mode_cfgs` (see **Overriding Rendering Mode Settings**
   below). Rendering Modes of Visualizers are set to performance by default.

   .. code-block:: python
      # Set KitVisualizer to balanced rendering mode
      sim_cfg = sim_utils.SimulationCfg(
          visualizer_cfgs=[
              KitVisualizerCfg(
                  rendering_mode="balanced",
              ),
          ],
      )

3. **Renderer Configs.** Similar to Visualizer Configs, from a Renderer Config, set the ``rendering_mode`` field to a profile
   name defined by the RenderingMode Configs. Rendering Modes of Renderers are set to None by default, which uses the native rendering settings of the workflow.

   .. code-block:: python
      # Set RTX Renderer to quality rendering mode.
      camera_cfg = CameraCfg(
          prim_path="/World/envs/env_.*/Camera",
          height=480,
          width=640,
          renderer_cfg=IsaacRtxRendererCfg(rendering_mode="quality"),
      )

.. note::
   If ``rendering_mode=None``, Isaac Lab uses the native rendering settings of the workflow

Example renders from the ``set_rendering_mode.py`` script.
To help assess rendering, the example scene includes reflections, translucency,
direct and ambient lighting, and several material types.

-  Quality Mode

   .. image:: ../_static/how-to/howto_rendering_example_quality.jpg
      :width: 100%
      :alt: Quality Rendering Mode Example

-  Balanced Mode

   .. image:: ../_static/how-to/howto_rendering_example_balanced.jpg
      :width: 100%
      :alt: Balanced Rendering Mode Example

-  Performance Mode

   .. image:: ../_static/how-to/howto_rendering_example_performance.jpg
      :width: 100%
      :alt: Performance Rendering Mode Example

Overriding Rendering Mode Settings
----------------------------------

Preset rendering settings can be overwritten via
:class:`~isaaclab.rendering_mode.RenderingModeCfg`.

The built-in ``rendering_mode_preset`` field only accepts ``performance``, ``balanced``, or ``quality``; those map to
fixed RTX baselines in ``isaaclab_physx.rendering.rtx_rendering_mode_presets`` (also available via :func:`isaaclab.rendering_mode.get_rendering_mode_preset`).

Rendering settings can be customized by either overwriting
specific settings of presets via ``kit_*`` fields (option 1) or by defining and adding a new **named profile** to :attr:`~sim.SimulationCfg.rendering_mode_cfgs` (option 2).

1. :class:`~isaaclab.rendering_mode.RenderingModeCfg` supports overwriting specific settings via explicit
   ``kit_*`` fields that map to underlying RTX settings.

   .. code-block:: python
      mode_cfg = RenderingModeCfg(
          rendering_mode_preset="performance",
          # explicit field overrides
          kit_enable_translucency=True,  # defaults to False in performance mode
          kit_enable_reflections=True,   # defaults to False in performance mode
          kit_dlss_mode=3,               # defaults to 0 in performance mode
      )

   List of Kit settings.

   .. table::
      :widths: 25 75

      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_translucency            | Bool. Enables translucency for specular transmissive surfaces such as   |
      |                                    | glass at the cost of some performance.                                  |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_reflections             | Bool. Enables reflections at the cost of some performance.              |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_global_illumination     | Bool. Enables Diffuse Global Illumination at the cost of some           |
      |                                    | performance.                                                            |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_antialiasing_mode              | Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"].                          |
      |                                    | DLSS boosts performance by reconstructing higher-resolution frames.      |
      |                                    | DLAA prioritizes image quality using the same SR technology as DLSS.    |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_dlssg                   | Bool. Enables DLSS-G frame generation (Ada Lovelace GPU required).      |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_dl_denoiser             | Bool. Enables DL denoiser (quality up, performance down).               |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_dlss_mode                      | Literal[0, 1, 2, 3] = Performance, Balanced, Quality, Auto.             |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_direct_lighting         | Bool. Enables direct light contributions from lights.                   |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_samples_per_pixel              | Int. Direct lighting samples-per-pixel (higher = better, slower).       |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_shadows                 | Bool. Enables shadows at performance cost.                              |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_ambient_occlusion       | Bool. Enables ambient occlusion at performance cost.                    |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_dome_light_upper_lower_strategy | Literal[0, 3, 4]. Maps to ``/rtx/domeLight/upperLowerStrategy`` (dome   |
      |                                    | light upper/lower hemisphere handling; see Omniverse RTX docs for     |
      |                                    | semantics of each value).                                             |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_disocclusion_scale             | Float. Aggressiveness of disocclusion handling for tiled rendering      |
      |                                    | (maps to ``/rtx/aovConverter/disocclusionScale``).                      |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_nre_compositing_renderer_hints | Int. NRE compositing hint (Isaac Lab apps use ``3``). Maps to           |
      |                                    | ``/omni/rtx/nre/compositing/rendererHints``.                            |
      +------------------------------------+-------------------------------------------------------------------------+


2. If you need a custom profile, define your own named entry in
   :attr:`~sim.SimulationCfg.rendering_mode_cfgs` and reference that name from ``KitVisualizerCfg.rendering_mode``
   (:mod:`isaaclab_physx.visualizers`) and/or ``camera_cfg.renderer_cfg.rendering_mode`` (RTX renderers under
   :mod:`isaaclab_physx.renderers`).

   .. code-block:: python
      sim_cfg = sim_utils.SimulationCfg(
          rendering_mode_cfgs={
              "my_profile": RenderingModeCfg(
                  rendering_mode_preset="balanced",
                  kit_enable_reflections=True,
                  kit_dlss_mode=2,
              ),
          },
          visualizer_cfgs=[KitVisualizerCfg(rendering_mode="my_profile")],
      )

      camera_cfg = CameraCfg(
          prim_path="/World/envs/env_.*/Camera",
          height=480,
          width=640,
          renderer_cfg=IsaacRtxRendererCfg(rendering_mode="my_profile"),
      )

Current Limitations
-------------------

For performance reasons, we default to using DLSS for denoising, which generally provides better performance.
This may result in renders of lower quality, which may be especially evident at lower resolutions.
Due to this, we recommend using per-tile or per-camera resolution of at least 100 x 100.
For renders at lower resolutions, we advice setting the ``antialiasing_mode`` attribute in :class:`~sim.RenderCfg` to
``DLAA``, and also potentially enabling ``enable_dl_denoiser``. Both of these settings should help improve render
quality, but also comes at a cost of performance. Additional rendering parameters can also be specified in :class:`~sim.RenderCfg`.


If you observe visual artifacts such as ghosting or disocclusion issues when using tiled rendering, you can try
adjusting the ``disocclusionScale`` parameter. This setting controls how aggressively the renderer handles
areas that become newly visible between frames:

.. code-block:: python

   render_cfg = sim_utils.RenderCfg(
      carb_settings={
         "/rtx/aovConverter/disocclusionScale": 10000,
      }
   )

.. note::

   This parameter is not commonly exposed as it may have side effects in certain scenarios.
   Only use it as a last resort if other quality settings do not resolve the visual artifacts.
   The value can be adjusted to a very high value to reduce disocclusion artifacts.


Rendering UsdVol 3D Gaussian Scenes in Multiple Environments
------------------------------------------------------------

When using UsdVol volumes with 3D Gaussian particles (e.g. exported from
`3DGRUT <https://github.com/nv-tlabs/3dgrut?tab=readme-ov-file#exporting-usdz-for-use-in-omniverse-and-isaac-sim>`_)
in **multiple environments**, you must set the following so the renderer uses the correct compositing path:

.. code-block:: python

   render_cfg = sim_utils.RenderCfg(
      carb_settings={
         "omni.rtx.nre.compositing.rendererHints": 3,
      }
   )

.. warning::

   With multiple environments, each environment holds its own copy of the scene, increasing device memory use,
   and environments are rendered one after another, which can substantially slow down rendering.
