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

``granular.usdc`` contains the approach, exit, and empty tray, not a frozen
snapshot of moving particles. Newton MPM and PhysX PBD particles, solver
settings, and backend-specific tray contacts remain in ``granular.py``.
The granular demo's existing sky-light configuration is separate from these
texture-free terrain assets.

To regenerate all three assets from the repository root::

    uv run python tools/terrains/export_g1_courses.py --output source/isaaclab_tasks/isaaclab_tasks/contrib/g1_terrain/assets

The exporter is an offline asset-authoring tool. The runtime task does not
import it and does not require PIL or temporary texture directories for terrain.
