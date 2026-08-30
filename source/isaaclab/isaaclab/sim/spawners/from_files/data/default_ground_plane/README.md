# Isaac Lab default ground plane

This directory contains the source asset for Isaac Lab's tiled default visual ground plane.

Asset contract:

- 100 m × 100 m visual mesh with an infinite USD collision plane;
- warm-white albedo with NVIDIA green (`#76B900`);
- 1 m primary lines and 5 m landmarks, with no minor subdivisions;
- seamless 5 m × 5 m texture tile repeated 20 times across the default plane at exactly 200 texels per meter;
- equal 10 mm landmark and primary lines, represented by two actual texels; both use exact NVIDIA green, while stronger emission and intersection nodes distinguish the 5 m landmarks;
- compatible prim paths for `spawn_ground_plane()`: `/World/Environment`, `/World/GroundPlane/CollisionPlane`, `/World/Looks/theGrid/Shader`, and `/World/SphereLight`.
- a single texture mapping, the mesh's face-varying `primvars:st`, where one uv unit is one 5 m tile; the OmniPBR shader reads that UV set with projection disabled and identity `texture_scale`.

The USD is self-contained apart from its three adjacent textures and Isaac Sim's built-in `OmniPBR.mdl`. It does not depend on the legacy `default_environment.usd` or its blue-grid textures.

## Renderer compatibility

The asset is bundled with the `isaaclab` package so the default does not require Nucleus access. Kit, Newton GL, and Newton Viewer RTX all resolve the texture through `primvars:st`, so the tiling has one source of truth rather than an OmniPBR projection and an equivalent UV set that can drift apart. Because the mapping lives in the mesh, scaling the plane also scales the tile: `spawn_ground_plane()` rewrites those UVs alongside the scale to keep the 5 m texture tile—and therefore the 1 m grid—metric in all three renderers. Referencing the USD directly and scaling it without that adjustment stretches the grid. Plane terrains bound the visual mesh to the environment grid (with a 100 m minimum) for stable UV precision; the USD collision plane remains infinite.

The compatibility behavior is scoped to this bundled default. Explicit ground-plane USDs, generated terrains, height fields, and mesh terrains remain unaffected.
