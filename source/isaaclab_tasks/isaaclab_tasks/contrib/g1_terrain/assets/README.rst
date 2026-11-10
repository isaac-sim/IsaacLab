Fixed G1 course assets
======================

``platforms.usdc``, ``ramps.usdc``, and ``granular.usdc`` contain the original
course geometry, static contacts, and solid-color materials. Each file is
self-contained: it references no textures or other USD files. Coordinates are
in meters, with Z up. The default prim is ``/Course``.

The tasks load these files through ``TerrainImporterCfg(terrain_type="usd")``.
Changing the asset location requires changing only ``usd_path``.

Platforms and ramps have an ``appearance`` variant set. The default,
``simulation``, exposes the original geometry to physics and the depth camera.
After the initial depth update, the demo selects ``visual`` to display the
inset ramp base. No geometry is generated at runtime. Select ``visual`` when
viewing the USD independently to match the rendered preview.

``granular.usdc`` contains the approach, exit, and empty tray for Newton MPM.
``granular_physx.usdc`` is a separate, self-contained PhysX course containing
the initial state of 116,222 gravel and 930,666 sand particles, their systems,
materials, and six tray contacts. These particles remain dynamic after loading;
the file is not a baked animation. This larger asset is stored with Git LFS.

Backend presets in ``terrain_env_cfg.py`` select the appropriate USD, Newton
MPM objects, solver coupling, control cadence, and SONIC gains before launch.
``ParticleTerrainImporter`` in ``terrain_env.py`` only enables PhysX GPU
simulation and state readback; it no longer creates particles, contacts, or
materials. PhysX still switches to host state buffers after the launcher selects
the CUDA device.
The granular demo's existing sky-light configuration is separate from these
texture-free terrain assets.

To regenerate the three static assets from the repository root::

    uv run python tools/terrains/export_g1_courses.py --output source/isaaclab_tasks/isaaclab_tasks/contrib/g1_terrain/assets

To regenerate the PhysX particle asset without launching Kit or a simulator::

    uv run --with physx-usd-schemas==25.11.2.72696984 python tools/terrains/export_g1_particles.py --output source/isaaclab_tasks/isaaclab_tasks/contrib/g1_terrain/assets

Both exporters are offline tools. The runtime task does not import them and
does not require PIL or temporary texture directories for terrain.
