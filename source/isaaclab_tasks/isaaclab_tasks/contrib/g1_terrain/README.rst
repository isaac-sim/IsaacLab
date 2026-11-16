Fixed G1 course assets
======================

``platforms.usdc``, ``ramps.usdc``, and ``granular.usdc`` contain the original
course geometry, static contacts, and solid-color materials. Each file is
self-contained: it references no textures or other USD files. Coordinates are
in meters, with Z up. The default prim is ``/Course``.

The tasks load these files through ``TerrainImporterCfg(terrain_type="usd")``
from ``{ISAACLAB_NUCLEUS_DIR}/Contrib/G1Terrain`` without a version subdirectory.
Assets are hosted separately, not packaged with the Python source. The existing
``ISAACSIM_ASSET_ROOT`` setting selects the asset server or a local mirror;
individual courses can also override ``env.scene.terrain.usd_path``.

Select the course with ``presets=platforms``, ``presets=ramps``, or
``presets=granular``. Python callers use ``G1PlatformsEnvCfg``, ``G1RampsEnvCfg``,
or ``G1GranularEnvCfg`` respectively. The ``terrain_name`` field has been removed;
replace ``G1PlatformsEnvCfg(terrain_name="ramps")`` with ``G1RampsEnvCfg()`` and
replace ``env.terrain_name=ramps`` with ``presets=ramps``. The demo commands
continue to select these presets through their existing arguments.

Platforms and ramps have an ``appearance`` variant set. The default,
``simulation``, exposes the original geometry to physics and the depth camera.
After the initial depth update, the demo selects ``visual`` to display the
inset ramp base. No geometry is generated at runtime. Select ``visual`` when
viewing the USD independently to match the rendered preview.

``granular.usdc`` contains the approach, exit, and empty tray for Newton MPM.
``granular_physx.usdc`` is a separate, self-contained PhysX course containing
the initial state of 116,222 gravel and 930,666 sand particles, their systems,
materials, and six tray contacts. These particles remain dynamic after loading;
the file is not a baked animation.

Backend presets in ``terrain_env_cfg.py`` select the appropriate USD, Newton
MPM objects, solver coupling, control cadence, and SONIC gains before launch.
``ParticleTerrainImporter`` in ``terrain_env.py`` only enables PhysX GPU
simulation and state readback; it no longer creates particles, contacts, or
materials. PhysX still switches to host state buffers after the launcher selects
the CUDA device.
The granular demo's existing sky-light configuration is separate from these
texture-free terrain assets.
