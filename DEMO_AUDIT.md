# Isaac Lab Demo Audit

This report reviews the 35 demos packaged for the incoming Isaac Lab 3.0 release. The target is a focused core
catalog of roughly 29 demos after specialized examples are relocated and overlapping examples are consolidated.

## Core showcases

| Demo | Recommendation | Notes |
| --- | --- | --- |
| `arl-robot-1` | Move to contrib | Specific robot and contrib controller; keep it beside its implementation rather than in the core catalog. |
| `arms` | Keep, consolidate | Useful asset-family gallery. Share its spawn/reset runner with the other robot galleries. |
| `bin-packing` | Keep, simplify | Good heterogeneous-cloning example. Separate layout generation from simulation control. |
| `bipeds` | Keep, consolidate | Valuable gallery with substantial overlap with the other robot-family demos. |
| `cables` | Keep | Focused Newton VBD capability with no equivalent elsewhere in the catalog. |
| `deformables` | Keep, improve | Important cross-backend comparison. Simplify backend configuration and add bounded execution. |
| `h1-locomotion` | Keep | Strong policy showcase. Continue isolating viewport interaction and testing command overrides. |
| `hands` | Keep, consolidate | Useful asset gallery; share the common robot-gallery implementation. |
| `haply-teleoperation` | Move to teleop | Hardware-specific example that should be owned and optionally registered by the teleoperation package. |
| `heterogeneous-scene` | Keep | Clearly demonstrates 3.0 scene composition without constructing task environments. |
| `markers` | Keep | Small, fundamental visualization example. Add bounded execution and complete typing. |
| `multi-asset` | Keep, improve | Useful heterogeneous-asset example. Clarify its distinction from `heterogeneous-scene`. |
| `newton-block-and-tackle` | Keep | Distinct interactive VBD showcase. |
| `newton-dominoes` | Keep | Distinct XPBD showcase and a useful packaged-asset example. |
| `pick-and-place` | Keep, refactor | Good surface-gripper showcase, but `InteractiveScene` would be leaner than a large `DirectRLEnv`. |
| `procedural-terrain` | Keep | Focused and useful. Add bounded execution and complete typing. |
| `quadcopter` | Fold into a vehicle gallery | The current asset/reset loop is too small to justify a standalone demo without meaningful flight control. |
| `quadrupeds` | Keep, consolidate | Valuable gallery; share the common robot-gallery implementation. |
| `visual-color-randomization` | Keep, improve | Good material-randomization showcase. Remove the remaining function-local import. |

## MPM showcases

| Demo | Recommendation | Notes |
| --- | --- | --- |
| `mpm-granular` | Keep | Best minimal entry point for Newton MPM. Complete typing and reduce avoidable local imports. |
| `mpm-two-way-coupling` | Keep | Demonstrates a distinct rigid-body/MPM interaction. |
| `snowball-smash` | Keep as showcase | Worth retaining as a polished coupling showcase; share setup utilities with the other MPM demos. |
| `teapot-fill` | Keep, split up | Flagship visual demo, but geometry, asset preparation, and animation helpers should leave the 800-line entry point. |

## Sensor showcases

| Demo | Recommendation | Notes |
| --- | --- | --- |
| `camera` | Keep | Useful Camera versus RayCasterCamera comparison with opt-in image output. |
| `contact-sensor` | Keep | Focused cross-backend contact example. |
| `frame-transformer` | Keep | Fundamental sensor example with appropriate scope. |
| `imu` | Keep, share runner | Important real-sensor abstraction; share robot reset/step code with PVA. |
| `multi-mesh-ray-caster` | Keep, clean up | Distinct dynamic-mesh capability. Remove placeholder prose, bound execution, and complete typing. |
| `multi-mesh-ray-caster-camera` | Keep, consolidate | Distinct projection API, but asset and scene construction should be shared with the non-camera demo. |
| `newton-raycast-heightfield` | Merge | Combine with moving geometry as `newton-raycast --scene heightfield`. |
| `newton-raycast-moving-geometry` | Merge | Combine with the height-field example and remove CPU/NumPy conversion helpers where direct interop works. |
| `ppisp-camera` | Move to renderer validation | This is a renderer comparison and QA workflow rather than a focused sensor demo. |
| `pva` | Keep, share runner | Useful distinction from IMU; share its common loop with the IMU demo. |
| `ray-caster` | Keep | Appropriate minimal lidar-style example. |
| `tactile-sensor` | Move to contrib | Experimental and asset-specific; it should be owned by the TacSL implementation package. |

## Recommended execution order

1. Move `arl-robot-1`, Haply, TacSL, and PPISP to their owning packages.
2. Fold `quadcopter` into a meaningful vehicle gallery or retire it.
3. Merge the two Newton raycast diagnostics.
4. Consolidate the robot galleries behind one small private runner while retaining useful public names.
5. Share only clearly duplicated sensor runners: IMU/PVA and the two multi-mesh raycasters.
6. Split `teapot-fill` and simplify `pick-and-place`.
7. Add bounded execution, complete type annotations, remove placeholder comments, and normalize status messages.
8. Require every public demo to pass `--help`, a bounded headless launch, wheel-content checks, and its declared backend matrix.
