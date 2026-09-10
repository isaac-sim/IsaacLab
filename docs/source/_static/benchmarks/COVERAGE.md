# Release benchmark coverage — September 9, 2026

## Result

**115 measurements across 36 of 44 eligible CORE tasks** are now used identically
for Release and Develop: **28 Collection, 87 Training**, including **34 camera
measurements**. September 9 supplied 76 measurements; 39 use the latest eligible
older measurement for their configuration. No single day had complete coverage.
The best daily coverage was 76 distinct series (including September 9).

This is a **release documentation baseline assembled from develop runs**, not
proof of performance on a release commit. Both channels intentionally use the
same CSV contents and snapshot date. The actual ingestion date, original
measurement timestamp, source record/entry, and measured commit are preserved.
Older values are not relabeled as measurements taken on release day.

| Source ingestion date | Selected measurements |
| --- | ---: |
| 2026-08-24 | 28 |
| 2026-08-26 | 4 |
| 2026-08-27 | 7 |
| 2026-09-09 | 76 |

The source measurement dates span August 23–September 9. Measurement timestamps
lack timezone information; only ingestion dates are treated as UTC. The August 24
batch includes one measurement timestamped August 23, and August 27 ingestions
include measurements timestamped August 26.

## Which environments lack default-count runs

**All plotted measurements use the configured environment default.** Runs at other
counts are not substituted. The flags below refer to usable results on the target
single-GPU hardware anywhere in the August 17–September 9 window, including older
fallback dates. A present run does not imply every backend/renderer is covered.

**34 environments lack default-count Collection runs; eight also lack default-count
Training runs.** The other 26 have default-count Training data only. The eight
missing both workloads are the six Franka cable/cloth/soft variants, Pendulum MARL
Direct, and manager-based Shadow Handover. See the exact names below.

[Download these flags](coverage-environment-defaults.csv). Alternative counts are
listed separately in that CSV and per backend/renderer in the configuration matrix.

| Environment | Default envs | Collection at default | Training at default |
| --- | ---: | --- | --- |
| Isaac-Ant-Direct | 4096 | MISSING | present |
| Isaac-Ant | 4096 | MISSING | present |
| Isaac-Cartpole-Direct | 4096 | MISSING | present |
| Isaac-Cartpole | 4096 | MISSING | present |
| Isaac-Cartpole-Camera-Direct | 512 | present | present |
| Isaac-Cartpole-Camera | 512 | present | present |
| Isaac-Humanoid-Direct | 4096 | MISSING | present |
| Isaac-Humanoid | 4096 | MISSING | present |
| Isaac-Lift-Cable-Franka | 8192 | MISSING | MISSING |
| Isaac-Lift-Cable-Franka-Camera | 128 | MISSING | MISSING |
| Isaac-Lift-Cloth-Franka | 2048 | MISSING | MISSING |
| Isaac-Lift-Cloth-Franka-Camera | 128 | MISSING | MISSING |
| Isaac-Lift-Franka | 4096 | MISSING | present |
| Isaac-Lift-KukaAllegro | 4096 | MISSING | present |
| Isaac-Lift-KukaAllegro-Camera | 4096 | present | present |
| Isaac-Lift-Soft-Franka | 2048 | MISSING | MISSING |
| Isaac-Lift-Soft-Franka-Camera | 128 | MISSING | MISSING |
| Isaac-Open-Drawer-Franka-Direct | 4096 | MISSING | present |
| Isaac-Open-Drawer-Franka | 4096 | MISSING | present |
| Isaac-Pendulum-MARL-Direct | 4096 | MISSING | MISSING |
| Isaac-Reach-Franka | 4096 | MISSING | present |
| Isaac-Reach-Franka-OSC | 4096 | MISSING | present |
| Isaac-Reach-UR10 | 4096 | MISSING | present |
| Isaac-Reorient-Cube-Allegro-Direct | 8192 | present | present |
| Isaac-Reorient-Cube-Allegro | 8192 | present | present |
| Isaac-Reorient-Cube-Shadow-Direct | 8192 | present | present |
| Isaac-Reorient-Cube-Shadow | 8192 | present | present |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 1225 | present | present |
| Isaac-Reorient-Cube-Shadow-Camera | 1225 | present | present |
| Isaac-Reorient-Franka | 4096 | MISSING | present |
| Isaac-Reorient-KukaAllegro | 4096 | MISSING | present |
| Isaac-Reorient-KukaAllegro-Camera | 4096 | present | present |
| Isaac-Shadow-Handover-Direct | 2048 | MISSING | present |
| Isaac-Shadow-Handover | 2048 | MISSING | MISSING |
| Isaac-Velocity-Flat-AnymalD | 4096 | MISSING | present |
| Isaac-Velocity-Flat-Cassie | 4096 | MISSING | present |
| Isaac-Velocity-Flat-G1 | 4096 | MISSING | present |
| Isaac-Velocity-Flat-H1 | 4096 | MISSING | present |
| Isaac-Velocity-Flat-UnitreeGo2 | 4096 | MISSING | present |
| Isaac-Velocity-Rough-AnymalD | 4096 | MISSING | present |
| Isaac-Velocity-Rough-Cassie | 4096 | MISSING | present |
| Isaac-Velocity-Rough-G1 | 4096 | MISSING | present |
| Isaac-Velocity-Rough-H1 | 4096 | MISSING | present |
| Isaac-Velocity-Rough-UnitreeGo2 | 4096 | MISSING | present |

## What is missing

The minimum comparison matrix contains **312 combinations**: task × supported
physics/renderer pair × Collection/Training, at each task's training-default
count. **88 are covered (28.2%); 224 are missing (71.8%).** Multiple domain
profiles account for the 115 plotted measurements; they do not increase the
number of covered pairs.

- **164 missing combinations have usable results at other environment counts**
  on the target GPU. They cannot fill default-count comparisons. Run them again
  at the count in the matrix; do not scale FPS or relabel the environment count.
- **60 combinations have no WARM entry on any queried machine** in this window.
  These include all configurations of the eight entirely absent tasks below,
  plus **12 OV PhysX + Newton Renderer gaps** (both workloads for each of
  the six otherwise measured camera tasks).
- Six camera tasks have explicit default-count coverage only for **Isaac Sim
  PhysX + Newton Renderer**. The Franka cable/cloth/soft camera tasks have none.
- Most non-camera Collection gaps have runs at 8,192 and 16,384 rather than
  the default count. OV PhysX Training gaps also need runs at the defaults. The exact
  available alternative counts are in the configuration CSV.

Entirely absent tasks (all machines, all counts, both workflows):

- `Isaac-Lift-Cable-Franka`
- `Isaac-Lift-Cable-Franka-Camera`
- `Isaac-Lift-Cloth-Franka`
- `Isaac-Lift-Cloth-Franka-Camera`
- `Isaac-Lift-Soft-Franka`
- `Isaac-Lift-Soft-Franka-Camera`
- `Isaac-Pendulum-MARL-Direct`
- `Isaac-Shadow-Handover`

`Isaac-Pendulum-MARL-Direct` needs RL-Games or SKRL training: it has no RSL-RL
registration. `Isaac-Fourbar-Pole-Swingup` supports only Kamino and is excluded
from the graph's scope, along with all other Kamino configurations. Contributed
tasks are outside this CORE benchmark baseline.

## Complete minimum coverage matrix

**P** = Isaac Sim PhysX, **N** = Newton MJWarp, **V** = Newton MJWarp + VBD,
**O** = OV PhysX. Renderers: **I** = Isaac Sim RTX, **W** = Newton Renderer,
**R** = OV RTX. A dash means no pair-level gaps, not complete domain-preset coverage.
The incompatible P/R and O/I pairs are excluded.

| CORE task | Default envs | Collection gaps | Training gaps |
| --- | ---: | --- | --- |
| Isaac-Ant-Direct | 4096 | P; N; O | O |
| Isaac-Ant | 4096 | P; N; O | O |
| Isaac-Cartpole-Direct | 4096 | P; N; O | O |
| Isaac-Cartpole | 4096 | P; N; O | O |
| Isaac-Cartpole-Camera-Direct | 512 | P/I; N/I; N/W; N/R; O/W; O/R | P/I; N/I; N/W; N/R; O/W; O/R |
| Isaac-Cartpole-Camera | 512 | P/I; N/I; N/W; N/R; O/W; O/R | P/I; N/I; N/W; N/R; O/W; O/R |
| Isaac-Humanoid-Direct | 4096 | P; N; O | O |
| Isaac-Humanoid | 4096 | P; N; O | O |
| Isaac-Lift-Cable-Franka | 8192 | V | V |
| Isaac-Lift-Cable-Franka-Camera | 128 | V/I; V/W; V/R | V/I; V/W; V/R |
| Isaac-Lift-Cloth-Franka | 2048 | P; V | P; V |
| Isaac-Lift-Cloth-Franka-Camera | 128 | P/I; P/W; V/I; V/W; V/R | P/I; P/W; V/I; V/W; V/R |
| Isaac-Lift-Franka | 4096 | P; N; O | O |
| Isaac-Lift-KukaAllegro | 4096 | P; N; O | O |
| Isaac-Lift-KukaAllegro-Camera | 4096 | P/I; N/I; N/W; N/R; O/W; O/R | P/I; N/I; N/W; N/R; O/W; O/R |
| Isaac-Lift-Soft-Franka | 2048 | P; V | P; V |
| Isaac-Lift-Soft-Franka-Camera | 128 | P/I; P/W; V/I; V/W; V/R | P/I; P/W; V/I; V/W; V/R |
| Isaac-Open-Drawer-Franka-Direct | 4096 | P; N; O | O |
| Isaac-Open-Drawer-Franka | 4096 | P; N; O | O |
| Isaac-Pendulum-MARL-Direct | 4096 | P; N; O | P; N; O |
| Isaac-Reach-Franka | 4096 | P; N; O | O |
| Isaac-Reach-Franka-OSC | 4096 | P; N; O | O |
| Isaac-Reach-UR10 | 4096 | P; N; O | O |
| Isaac-Reorient-Cube-Allegro-Direct | 8192 | — | — |
| Isaac-Reorient-Cube-Allegro | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow-Direct | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 1225 | P/I; N/I; N/W; N/R; O/W; O/R | P/I; N/I; N/W; N/R; O/W; O/R |
| Isaac-Reorient-Cube-Shadow-Camera | 1225 | P/I; N/I; N/W; N/R; O/W; O/R | P/I; N/I; N/W; N/R; O/W; O/R |
| Isaac-Reorient-Franka | 4096 | P; N; O | O |
| Isaac-Reorient-KukaAllegro | 4096 | P; N; O | O |
| Isaac-Reorient-KukaAllegro-Camera | 4096 | P/I; N/I; N/W; N/R; O/W; O/R | P/I; N/I; N/W; N/R; O/W; O/R |
| Isaac-Shadow-Handover-Direct | 2048 | P; N; O | O |
| Isaac-Shadow-Handover | 2048 | P; N; O | P; N; O |
| Isaac-Velocity-Flat-AnymalD | 4096 | P; N; O | O |
| Isaac-Velocity-Flat-Cassie | 4096 | P; N; O | O |
| Isaac-Velocity-Flat-G1 | 4096 | P; N; O | O |
| Isaac-Velocity-Flat-H1 | 4096 | P; N; O | O |
| Isaac-Velocity-Flat-UnitreeGo2 | 4096 | P; N; O | O |
| Isaac-Velocity-Rough-AnymalD | 4096 | P; N; O | O |
| Isaac-Velocity-Rough-Cassie | 4096 | P; N; O | O |
| Isaac-Velocity-Rough-G1 | 4096 | P; N; O | O |
| Isaac-Velocity-Rough-H1 | 4096 | P; N; O | O |
| Isaac-Velocity-Rough-UnitreeGo2 | 4096 | P; N; O | O |

## Preset and camera-profile coverage

Pair-level coverage only requires one measured profile. It does **not** establish
coverage for every image type, resolution, camera count, object set, controller,
feature encoder, randomization preset, or RL library.

- [Observed profiles](coverage-observed-profiles.csv) lists every selected
  physics/renderer/domain/resolution profile and whether Collection and Training
  are both present. Of 87 observed profiles, 28 have both workloads and 59
  lack Collection. Missing counterparts need a run with the same profile.
- [Registered preset selectors](coverage-preset-selectors.csv) enumerates every
  advertised individual domain selector against the pair/workload matrix and
  identifies whether it occurs explicitly in a selected measurement. “Not
  explicitly recorded” can also mean an implicit default; it is not evidence
  that the configuration failed or that the selector is compatible with every
  backend. This is a collection checklist, not a validated Cartesian test matrix.
- The recovered Kuka camera profiles explicitly include cube, single/dual camera,
  and RGB64/depth64 combinations. They do not establish coverage for all the
  advertised 128/256, albedo, segmentation, shading, ray-caster, or shapes variants.
- All **34 selected camera measurements lack resolved camera dimensions** in
  their metadata. Some preset names encode a requested resolution, but the export
  does not invent verified dimensions from a name. **Six** also lack an explicit
  domain profile. Preserve these measurements as recorded, and rerun with resolved
  sensor width/height, image types, camera count, and complete domain metadata
  before claiming fully matched camera comparisons.

## Incorrect, ambiguous, or weak source data

[Data issues](coverage-data-issues.csv) contains exact record IDs and WARM entry
keys for every detected count/backend-label conflict or missing physics/renderer on the
target single-GPU hardware in the queried window.

1. **3,234 camera WARM entries have no renderer in the database renderer field.**
   Some have an explicit requested renderer in session presets; that is retained
   in the audit CSV but is not substituted for measured renderer identity.
   The previous default-count selection included three such latest entries:
   record **165741** (Cartpole Camera and Camera Direct) and **165752**
   (Reorient KukaAllegro Camera). They are excluded from both graph datasets.
   Fix the recorder to emit the resolved renderer for each WARM entry and rerun
   or verify its original artifacts before assigning a renderer.
2. **364 WARM entries have comparison-group physics and environment-count
   labels that disagree with the recorded configuration.** For example, record
   **153798** is grouped as `OVPHYSX_E8192`, but its WARM Cartpole camera entries
   record Isaac Sim PhysX + Newton Renderer with 512 environments. **28 selected
   measurements** have these misleading group labels; their graph labels follow
   the mutually consistent recorded backend and task metadata. The entry key and `benchmark_info.num_envs` agree
   in the inspected target-hardware WARM data, as does `runtime.num_envs`.
   Use measured counts; correct group labeling in the producer. The audit CSV
   preserves the group name, recorded physics, requested preset, and measured count.
3. **6,815 WARM entries lack a database physics backend.** Requested session
   presets are available, but the export does not substitute a requested backend
   for missing recorded identity. These entries are excluded. Record resolved
   physics per WARM entry, or verify original artifacts before restoring them.
4. **All 115 selected measurements report a dirty working tree.** Their recorded
   commits identify the base source, but the database does not establish the
   local diff. This does not prove incorrect FPS. For a reproducible release
   baseline, run a clean release checkout or archive the exact patch and
   dependency lock alongside the results.
5. **22 selected measurements have FPS standard deviation above 25% of the
   mean.** They remain measured values, not proven failures; per-row quality
   notes flag them. Training entries use only 10 measured iterations. Repeat
   with a longer fixed measurement window and multiple seeds/runs before making
   precise performance claims. The CSV retains mean, standard deviation, peak,
   and iteration count.
6. **39 selected values predate release day**, including all selected camera
   data. Record dates and commits are visible/preserved; they provide provisional
   coverage, not evidence that camera performance is unchanged at the release.

No nonpositive total FPS, runtime/benchmark count disagreement, or missing/ambiguous
successful task metadata was found among the selected measurements. Positive
training throughput is not proof of policy convergence or task correctness.

## Collection plan

1. Fill the 164 count mismatches using the exact default counts in
   [the configuration matrix](coverage-configurations.csv), keeping the same
   single-GPU machine and separate runtime/training workflows.
2. Schedule the 60 combinations with no WARM entry, starting with the eight
   entirely absent tasks; use a supported training library for Pendulum MARL.
3. Replace the older camera fallbacks with release-commit runs that record resolved
   physics, renderer, sensor dimensions, image types, complete presets, seed,
   actual environment count, world size, dependency versions, and source identity.
4. Fill the missing workload counterpart for each observed profile, then explicitly
   choose compatible camera/domain profiles from the registered-selector checklist.
5. Repeat noisy measurements and archive clean-source provenance. Re-export one
   snapshot to both channels until separate develop history is intentionally resumed.

## Query scope and evidence

Queried read-only on September 10, 2026: **61,867 database records** with
`2026-08-17 <= created_at < 2026-09-10`, across all four machine kinds in
`omni_runtime_isaac_lab_v3`. Expanded WARM entries produced 57,130 relevant CORE
runtime/training entries across all machines. Today's incomplete batch is excluded.
No claim is made about data outside this interval or failed runs lacking WARM KPIs.
“No WARM entry” does not distinguish an unscheduled task from a failed attempt.

Exports select `XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`, `num_gpus=1`, world size 1,
successful entries, finite positive mean total FPS, matching measured counts,
unambiguous successful WARM task metadata, and supported explicit camera renderers.
Defaults were checked against current CORE configuration source (not play overrides).
See [selection and provenance](README.md) for the export contract.
