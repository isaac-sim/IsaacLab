# ManipulationNet cable-management assets

This directory contains offline-derived USD assets for the ManipulationNet Cable Management Benchmark:

- `board.usdc`: the four official task-board segment meshes as render geometry and one `0.30 x 0.40 x 0.00635 m` active-workspace cube collider.
- `round_peg.usdc`: the official F1 round-peg mesh and a three-cylinder compound collider for its `9.5 mm` waist and `25 mm` flanges.

The dense triangle meshes have no `CollisionAPI`. Only the guide-purpose primitive proxies collide, keeping the assets inexpensive for MJWarp/VBD coupling while preserving the printed appearance. Both USD files are self-contained, meter-scale, Z-up, kinematic rigid components.

## Provenance

- Source: <https://github.com/ManipulationNet/mnet_client/tree/ros_2/assets/cable_management/cad_files>
- Branch: `ros_2`
- Pinned repository commit: `2745ccc6099fb3b65e89cbdbaf7af6521bf8dd29`
- Introducing CAD commit: `4e255a744cc230d24c43260947fb895b889051a8`
- Upstream license: Apache License 2.0; a copy is included as `LICENSE`.
- Source units: the STL format is unitless. The upstream board dimensions establish that one authored unit is one centimeter, so the conversion scale is `0.01` meters per source unit.

Pinned source hashes:

| Source file | SHA-256 |
|---|---|
| `board_segment_upper_left.stl` | `fa90f8e015401c743b9dd967166023e66c14b8883d9808e0675a915072a9442f` |
| `board_segment_upper_right.stl` | `fa90f8e015401c743b9dd967166023e66c14b8883d9808e0675a915072a9442f` |
| `board_segment_bottom_left.stl` | `1256f953cd5a9e18000f107310b265ed63b6a984252413c4be5a427f9a097585` |
| `board_segment_bottom_right.stl` | `6de8c5362d04f6a99a15b00f7655c5d706112103e9a5c8546f0e5306253be62c` |
| `round_peg.stl` | `29d8169aaf13374e7f3ebcbba5f85ef95592408498315686483a9c62b87230e7` |

Current derived artifact hashes (bitwise reproducible with the checked-out toolchain):

| Derived file | SHA-256 |
|---|---|
| `board.usdc` | `4c8056e1826857dbfd04eee69d407b12ce2fccaf3acb703d138f04c09b2472ca` |
| `round_peg.usdc` | `aa25d088c1e4e339664e22f58f4998cef306c96bb7413786db802ac07ab79d7c` |

## Rebuilding

Clone or check out the pinned upstream revision, then run:

```bash
uv run python scripts/tools/generate_manipulationnet_cable_assets.py \
  /path/to/mnet_client/assets/cable_management/cad_files
```

The generator validates every input hash, recenters the arbitrary CAD pivots, assembles the documented `400 x 300 mm` active board in the environment's `(X=300 mm, Y=400 mm)` frame, and writes both assets atomically. The lower panels retain their official `30 mm` front branding strips, so the render envelope extends to `X=-0.18 m`; that strip is intentionally outside the active-workspace collision cube.

The alignment clips are not included because their exact installed poses are not specified by the machine-readable CAD or benchmark coordinate definition. They are cosmetic for the fixed simulation board and do not change its active collision surface.
