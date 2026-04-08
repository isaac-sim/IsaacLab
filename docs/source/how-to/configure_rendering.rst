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

Rendering mode is selected as follows:

1. **Camera sensors.** Classic and tiled cameras do not define ``rendering_mode`` on the camera config itself; set
   it on the nested :attr:`~sensors.camera.camera_cfg.CameraCfg.renderer_cfg` (a :class:`~isaaclab.renderers.RendererCfg`).
   Use the same profile names as keys in :attr:`~sim.SimulationCfg.rendering_mode_cfgs`. When the CLI does not set
   an explicit mode, this field selects the profile; otherwise the CLI wins (see item 3 below). For Kit-style
   renderers (``default``, ``isaac_rtx``, ``rtx``), the profile applies RTX settings; for Newton Warp tiled cameras,
   configure :class:`~isaaclab_newton.renderers.newton_warp_renderer_cfg.NewtonWarpRendererCfg` directly instead.

2. **Kit visualizer.** On :class:`~isaaclab_physx.visualizers.kit_visualizer_cfg.KitVisualizerCfg`, set ``rendering_mode``
   to an entry from :attr:`~sim.SimulationCfg.rendering_mode_cfgs` (RTX / viewport settings).

   .. code-block:: python

      import isaaclab.sim as sim_utils
      from isaaclab_physx.visualizers import KitVisualizerCfg

      sim_cfg = sim_utils.SimulationCfg(
          visualizer_cfgs=[
              KitVisualizerCfg(
                  rendering_mode="performance",
              ),
          ],
      )

3. Use the ``--rendering_mode`` CLI argument, which takes precedence over
   ``camera_cfg.renderer_cfg.rendering_mode`` and over ``visualizer_cfg.rendering_mode``.

   .. code-block:: bash

      ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py --rendering_mode {performance/balanced/quality}

Notes:

* If ``rendering_mode=None`` on ``renderer_cfg`` and the CLI did not set an explicit mode, no RTX profile
  is applied for Kit-style camera renderers; other renderer types ignore ``rendering_mode``.
* If ``rendering_mode=None`` on a Kit visualizer, Isaac Lab does not apply RTX profile overrides
  (USD-authored / native defaults apply). Other visualizers ignore ``rendering_mode``; use their own config classes
  (for example :class:`~isaaclab_visualizers.newton.newton_visualizer_cfg.NewtonVisualizerCfg` for Newton GL).
* ``--rendering_mode`` is the supported CLI entry point.

.. note::

   :class:`~isaaclab.rendering_mode.RenderingModeCfg` only carries ``kit_*`` fields and built-in
   ``rendering_mode_preset`` values (``performance``, ``balanced``, ``quality``). Those profiles drive **Kit / RTX**
   carb settings for the Kit viewport and for Kit-style camera renderers (``default``, ``isaac_rtx``, ``rtx``).
   Newton GL and Newton Warp are configured on :class:`~isaaclab_visualizers.newton.newton_visualizer_cfg.NewtonVisualizerCfg`
   and :class:`~isaaclab_newton.renderers.newton_warp_renderer_cfg.NewtonWarpRendererCfg` respectively, not via
   ``SimulationCfg.rendering_mode_cfgs``.

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
fixed RTX baselines in ``isaaclab.rendering_mode.rendering_mode_presets``. Isaac Lab does not provide a supported
way to register additional preset baselines. Customization is done by adding a **named profile** to
:attr:`~sim.SimulationCfg.rendering_mode_cfgs` that picks one of the three baselines and overrides it with ``kit_*``
fields—see item 2 below.

There are two ways to provide settings that overwrite presets:

1. :class:`~isaaclab.rendering_mode.RenderingModeCfg` supports overwriting specific settings via explicit
   ``kit_*`` fields that map to underlying RTX settings.

   .. code-block:: python

      import isaaclab.sim as sim_utils
      from isaaclab.rendering_mode import RenderingModeCfg

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


2. If you need a custom profile, define your own named entry in
   :attr:`~sim.SimulationCfg.rendering_mode_cfgs` and reference that name from
   ``camera_cfg.renderer_cfg.rendering_mode`` (Kit-style renderers) and/or ``KitVisualizerCfg.rendering_mode``.

   .. code-block:: python

      import isaaclab.sim as sim_utils
      from isaaclab.rendering_mode import RenderingModeCfg
      from isaaclab_physx.visualizers import KitVisualizerCfg

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

Current Limitations
-------------------

For performance reasons, we default to using DLSS for denoising, which generally provides better performance.
This may result in renders of lower quality, which may be especially evident at lower resolutions.
Due to this, we recommend using per-tile or per-camera resolution of at least 100 x 100.
For renders at lower resolutions, we advise setting
``kit_antialiasing_mode="DLAA"`` in
:class:`~isaaclab.rendering_mode.RenderingModeCfg`, and potentially enabling
``kit_enable_dl_denoiser=True``. Both settings can improve quality at a cost
of performance.


If you observe visual artifacts such as ghosting or disocclusion issues when using tiled rendering, you can try
adjusting the ``disocclusionScale`` parameter. This setting controls how aggressively the renderer handles
areas that become newly visible between frames:

.. note::

   Low-level carb rendering settings (for example,
   ``/rtx/aovConverter/disocclusionScale``) are not currently exposed through
   :class:`~isaaclab.rendering_mode.RenderingModeCfg`.

.. note::

   This parameter is not commonly exposed as it may have side effects in certain scenarios.
   Only use it as a last resort if other quality settings do not resolve the visual artifacts.
   The value can be adjusted to a very high value to reduce disocclusion artifacts.


Rendering UsdVol 3D Gaussian Scenes in Multiple Environments
------------------------------------------------------------

When using UsdVol volumes with 3D Gaussian particles (e.g. exported from
`3DGRUT <https://github.com/nv-tlabs/3dgrut?tab=readme-ov-file#exporting-usdz-for-use-in-omniverse-and-isaac-sim>`_)
in **multiple environments**, you must set the following so the renderer uses the correct compositing path:

.. note::

   This setting is not currently exposed through
   :class:`~isaaclab.rendering_mode.RenderingModeCfg`.

.. warning::

   With multiple environments, each environment holds its own copy of the scene, increasing device memory use,
   and environments are rendered one after another, which can substantially slow down rendering.
