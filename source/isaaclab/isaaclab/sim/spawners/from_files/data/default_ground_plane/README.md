# Isaac Lab default ground plane

This directory contains the source asset for Isaac Lab's tiled default visual ground plane.

Asset contract:

- 100 m × 100 m visual mesh with an infinite USD collision plane;
- warm-white albedo with NVIDIA green (`#76B900`);
- 1 m primary lines and 5 m landmarks, with no minor subdivisions;
- seamless 5 m × 5 m texture tile repeated 20 times across the default plane at exactly 200 texels per meter;
- equal 10 mm landmark and primary lines, represented by two actual texels; both use exact NVIDIA green, while stronger emission and intersection nodes distinguish the 5 m landmarks;
- compatible prim paths for `spawn_ground_plane()`: `/World/Environment`, `/World/GroundPlane/CollisionPlane`, `/World/Looks/theGrid/Shader`, and `/World/SphereLight`.
- object-projected OmniPBR mapping for Kit renderers and equivalent authored UVs for renderers that consume mesh UVs.

The USD is self-contained apart from its three adjacent textures and Isaac Sim's built-in `OmniPBR.mdl`. It does not depend on the legacy `default_environment.usd` or its blue-grid textures.

## Renderer compatibility

The asset is bundled with the `isaaclab` package so the default does not require Nucleus access. Kit uses the authored OmniPBR projection, while Newton GL and Newton Viewer RTX consume the mesh UVs. When the bundled plane is resized, `spawn_ground_plane()` adjusts those UVs to keep the 5 m texture tile—and therefore the 1 m grid—metric in all three renderers. Plane terrains bound the visual mesh to the environment grid (with a 100 m minimum) for stable UV precision; the USD collision plane remains infinite.

The compatibility behavior is scoped to this bundled default. Explicit ground-plane USDs, generated terrains, height fields, and mesh terrains remain unaffected.
