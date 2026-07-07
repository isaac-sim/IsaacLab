# SimReady Asset Report: 2584N111 DisplayPort Cord Socket

## Deliverables

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| SimReady USD | `04_author_simready/2584n111_displayport_cord_socket_screws_removed_simready.usd` | `f2bd60706466dd0829b9e51720f54ddcfb22ccd24f8aae8153d69823995a8f07` |
| RTX render | `06_render_final/final.png` | `99eaddc80a75e38a5f37d5ca5fe30da5012d77c639890a800ca4ad198ea8a760` |
| Authoring report | `04_author_simready/authoring_report.json` | See bundle |
| Validation reports | `05_validate_final/` | See bundle |

![Final RTX render](06_render_final/final.png)

## Source Asset

- Input CAD: `/Users/lingq/Downloads/2584N111_Displayport Cord_socket_screws_removed.step`
- Input STEP SHA-256: `70069cb4791fca4e607492ab9aee63b83ccfefa50591879f339a81516360e967`
- Processing host: `horde@horde-dgxc.nvidia.com`
- Tooling: installed NVIDIA Skill Hub CAD-to-SimReady/OpenUSD validation tooling on the Horde machine

## USD Structure

| Property | Value |
| --- | --- |
| Default prim | `/tn__2584N111_Displayport_Cord_socket_screws_removed_` |
| Stage units | `metersPerUnit = 1.0` |
| Up axis | `Z` |
| Mesh count | `6` |
| Rigid body count | `6` |
| Collider count | `6` |
| Authored references | `0` |
| Prototype count | `0` |
| Final USD size | `413,443 bytes` |

The delivered USD was flattened to a single self-contained layer and the internal CAD prototype/reference structure was removed before final authoring.

## Geometry

| Metric | Value |
| --- | --- |
| Bounds min, meters | `(0.0, -0.007400000256299972, -0.018999323284626005)` |
| Bounds max, meters | `(0.03750000028610229, 0.007400000256299972, 0.01900067676305771)` |
| Bounds size, meters | `(0.03750000028610229, 0.014800000512599944, 0.03800000004768371)` |
| Point count | `13,828` |
| Triangle count | `12,653` |

## Materials

| CAD body | Visual material | Role |
| --- | --- | --- |
| `Body8` | Warm off-white plastic | Molded housing and rear strain-relief geometry |
| `Body4`, `Body5`, `Body6` | Brushed dark metal | DisplayPort socket shell/contact structures |
| `Body12`, `Body13` | Dark screw-bore metal | Screw-bore insert surfaces after screw removal |

All visual materials are authored as `UsdPreviewSurface` materials.

## Physical Properties

| Property | Value |
| --- | --- |
| Total estimated mass | `0.013 kg` |
| Collision approximation | `convexHull` per mesh |
| Static friction | `0.85` |
| Dynamic friction | `0.65` |
| Restitution | `0.08` |
| Physics density | `1200 kg/m^3` |

Mass was estimated from the small connector scale and distributed across the functional CAD bodies:

| Body | Mass, kg |
| --- | ---: |
| `Body4` | `0.0015` |
| `Body5` | `0.0020` |
| `Body6` | `0.0010` |
| `Body8` | `0.0075` |
| `Body12` | `0.0005` |
| `Body13` | `0.0005` |

## Grasp Annotation

| Property | Value |
| --- | --- |
| Path | `/tn__2584N111_Displayport_Cord_socket_screws_removed_/GraspAnnotations/grasp_identifier_01` |
| Type | `parallel_gripper_grasp_axis` |
| Target | Main plastic housing |
| Point A, meters | `(0.01875000074505806, -0.006599999964237213, 0.0)` |
| Point B, meters | `(0.01875000074505806, 0.006599999964237213, 0.0)` |
| Width | `0.0012 m` |

## Validation Summary

| Gate | Status | Errors | Failures | Warnings |
| --- | --- | ---: | ---: | ---: |
| Minimum USD validation | `PASS` | `0` | `0` | `1` |
| Omni asset validation | `PASS` | `0` | `0` | `7` |
| Omni geometry validation | `PASS` | `0` | `0` | `7` |
| Omni physics validation | `PASS` | `0` | `0` | `0` |
| SimReady profile validation | `PASS` | `0` | `0` | `0` |
| OVRTX render | `PASS` | `0` | `0` | `1` |

SimReady validation target:

- Profile: `Prop-Robotics-Neutral`
- Version: `1.0.0`
- Result: `PASS`
- SimReady issue counts: `ERROR=0`, `FAILURE=0`, `WARNING=0`

Passed SimReady features:

- `FET000_CORE`
- `FET001_BASE_NEUTRAL`
- `FET003_BASE_NEUTRAL`
- `FET004_BASE_NEUTRAL`
- `FET005_BASE_NEUTRAL`
- `FET006_BASE_MDL`

## Residual Non-Blocking Warnings

The final asset passed validation. Remaining warnings are non-blocking:

- `IndexedPrimvarChecker`: repeated `primvars:st` values can be indexed on the six CAD meshes.
- `ManifoldChecker`: `Body13/Mesh` has 2 non-manifold vertices from the source CAD tessellation.
- Render report warning: Pillow was unavailable in the validation virtualenv, so pixel inspection was skipped by that wrapper. The render file was produced successfully and manually inspected.
- Minimum USD warning: the validator reported an anonymous session layer while opening the USD; the delivered USD itself has no authored references and no prototypes.

## Conclusion

`2584n111_displayport_cord_socket_screws_removed_simready.usd` is SimReady for the validated `Prop-Robotics-Neutral` profile. It has explicit materials, physical material parameters, rigid bodies, convex-hull colliders, mass properties, stage units, up-axis metadata, and a grasp-axis annotation. Physics and SimReady profile validation are clean passes.
