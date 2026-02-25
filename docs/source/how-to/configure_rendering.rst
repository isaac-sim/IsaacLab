Configuring Rendering Settings
==============================

Isaac Lab offers three preset rendering mode profiles: ``performance``, ``balanced``, and ``quality``.
You can select a profile via CLI or from script config, then override specific Kit/Newton settings as needed.

Selecting a Rendering Mode Profile
----------------------------------

Rendering mode can be selected in two ways:

1. Set the visualizer profile selector field ``rendering_mode``, which selects an entry from
   :attr:`~sim.SimulationCfg.rendering_mode_cfgs`.

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

2. Use the ``--rendering_mode`` CLI argument, which takes precedence over
   ``visualizer_cfg.rendering_mode``.

   .. code-block:: bash

      ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py --rendering_mode {performance/balanced/quality}

Notes:

* If ``rendering_mode=None`` for a visualizer, Isaac Lab does not apply rendering overrides
  for that visualizer, and backend/native defaults (for Kit, USD-authored settings) are used.
* ``--rendering_mode`` is the supported CLI entry point.

Example renders from the ``set_rendering_mode.py`` script.
To help assess rendering, the example scene includes reflections, translucency, direct and ambient lighting, and several material types.

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

Overwriting Specific Rendering Settings
---------------------------------------

Preset rendering settings can be overwritten via :class:`~sim.RenderingModeCfg`.

There are two ways to provide settings that overwrite presets:

1. :class:`~sim.RenderingModeCfg` supports overwriting specific settings via explicit
   ``kit_*`` fields that map to underlying RTX settings.

   .. code-block:: python

      import isaaclab.sim as sim_utils

      mode_cfg = sim_utils.RenderingModeCfg(
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

2. Customize or add named profiles in :attr:`~sim.SimulationCfg.rendering_mode_cfgs`,
   then select them from per-visualizer profile selector fields (``rendering_mode``).

Current Limitations
-------------------

For performance reasons, DLSS-centric settings are commonly used by default.
At lower resolutions, quality artifacts may be more visible.
For low-resolution renders, consider:

* ``kit_antialiasing_mode="DLAA"``
* ``kit_enable_dl_denoiser=True``

in :class:`~sim.RenderingModeCfg`. These can improve quality at a performance cost.
