# Benchmark coverage — paired Collection and Training FPS

## Corrected interpretation

**Collection and Training come from the same WARM training record.** Collection
uses `Mean Collection FPS` (environment stepping plus policy inference); Training
uses `Mean Total FPS` (including policy updates). Both retain their own standard
deviation and peak. A standalone runtime benchmark is not required.

The previous report's “34 environments missing Collection” described standalone
runtime coverage and was misleading for these graph modes. It is superseded by
this report. Every selected training record has both positive FPS metrics, so
Collection and Training now have identical configuration/date coverage.

## Snapshot coverage

| Snapshot | Training records, each containing both metrics | Environments | Covered backend/renderer pairs |
| --- | ---: | ---: | ---: |
| August 28 | 64 | 30 / 44 | 64 / 156 |
| Current (September 9 baseline) | 87 | 36 / 44 | 70 / 156 |

Release contains only the 87 current records (**174 metric points**, 87 in each
mode). Develop contains 151 records (**302 metric points**, 151 in each mode):
August 28 plus the identical current records. August 14 was removed from the
graph because there are no qualifying runs.

The table displays only the latest FPS per configuration, rounded to an integer,
with backend names only. Dates and full configurations remain in tooltips; the
legend appears above the graph. Historical results remain visible in the graph.

Historical snapshots require the exact ingestion date. The current baseline keeps
the latest eligible run through September 9: 64 records ingested September 9,
14 on August 24, two on August 26, and seven on August 27. Older camera values
remain provisional; snapshot date is not the actual measurement date.

## Why August 14 has no points

The database was queried for August 14, including all machines. There were no
CORE training measurements at the configured default counts. On the target GPU,
training records covered only:

| Environment | Measured counts | Required default |
| --- | --- | ---: |
| Isaac-Lift-KukaAllegro-Camera | 8,192; 16,384 | 4,096 |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 8,192; 16,384 | 1,225 |
| Isaac-Velocity-Rough-G1 | 16,384 | 4,096 |

August 14 is omitted from the Develop graph and table.
Historical values are not backfilled from later dates or borrowed from other
counts. Lines do not bridge an explicitly missing intermediate snapshot.

## Why OV PhysX appears only for some environments

OV PhysX data exists. On the queried single-GPU machine its training measurements
use **8,192 or 16,384 environments**, not 4,096. The four cube-reorientation tasks
below default to 8,192, so OV PhysX is included for both metrics in their current
and August 28 tables:

- `Isaac-Reorient-Cube-Allegro`
- `Isaac-Reorient-Cube-Allegro-Direct`
- `Isaac-Reorient-Cube-Shadow`
- `Isaac-Reorient-Cube-Shadow-Direct`

Most other tasks default to 4,096; Cartpole cameras default to 512, Shadow cameras
to 1,225, and handover to 2,048. Their OV PhysX runs cannot be plotted as matching
default-count comparisons. Missing supported pairs now have an explicit
**Not available** value in the FPS table, with the reason in a tooltip.

Some older records have null database backend/renderer columns but explicit
selectors in `benchmark_info.presets` or the matching successful WARM task preset.
These records are recovered using that per-run evidence, with the source recorded
in the CSV. Backend identity is never inferred from comparison-group labels or
an implicit default. This recovers OV PhysX's August 28 default-count records.

## Which environments lack default-count runs

The flags below apply to **both Collection and Training**. “present” means at
least one backend/renderer profile has a usable paired record; it does not mean
all backends or all domain presets are covered. All plotted rows use the default
count; alternative counts remain audit information only.

| Environment | Default envs | Aug 28 | Current |
| --- | ---: | --- | --- |
| Isaac-Ant-Direct | 4096 | present | present |
| Isaac-Ant | 4096 | present | present |
| Isaac-Cartpole-Direct | 4096 | present | present |
| Isaac-Cartpole | 4096 | present | present |
| Isaac-Cartpole-Camera-Direct | 512 | MISSING | present |
| Isaac-Cartpole-Camera | 512 | MISSING | present |
| Isaac-Humanoid-Direct | 4096 | present | present |
| Isaac-Humanoid | 4096 | present | present |
| Isaac-Lift-Cable-Franka | 8192 | MISSING | MISSING |
| Isaac-Lift-Cable-Franka-Camera | 128 | MISSING | MISSING |
| Isaac-Lift-Cloth-Franka | 2048 | MISSING | MISSING |
| Isaac-Lift-Cloth-Franka-Camera | 128 | MISSING | MISSING |
| Isaac-Lift-Franka | 4096 | present | present |
| Isaac-Lift-KukaAllegro | 4096 | present | present |
| Isaac-Lift-KukaAllegro-Camera | 4096 | MISSING | present |
| Isaac-Lift-Soft-Franka | 2048 | MISSING | MISSING |
| Isaac-Lift-Soft-Franka-Camera | 128 | MISSING | MISSING |
| Isaac-Open-Drawer-Franka-Direct | 4096 | present | present |
| Isaac-Open-Drawer-Franka | 4096 | present | present |
| Isaac-Pendulum-MARL-Direct | 4096 | MISSING | MISSING |
| Isaac-Reach-Franka | 4096 | present | present |
| Isaac-Reach-Franka-OSC | 4096 | present | present |
| Isaac-Reach-UR10 | 4096 | present | present |
| Isaac-Reorient-Cube-Allegro-Direct | 8192 | present | present |
| Isaac-Reorient-Cube-Allegro | 8192 | present | present |
| Isaac-Reorient-Cube-Shadow-Direct | 8192 | present | present |
| Isaac-Reorient-Cube-Shadow | 8192 | present | present |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 1225 | MISSING | present |
| Isaac-Reorient-Cube-Shadow-Camera | 1225 | MISSING | present |
| Isaac-Reorient-Franka | 4096 | present | present |
| Isaac-Reorient-KukaAllegro | 4096 | present | present |
| Isaac-Reorient-KukaAllegro-Camera | 4096 | MISSING | present |
| Isaac-Shadow-Handover-Direct | 2048 | present | present |
| Isaac-Shadow-Handover | 2048 | MISSING | MISSING |
| Isaac-Velocity-Flat-AnymalD | 4096 | present | present |
| Isaac-Velocity-Flat-Cassie | 4096 | present | present |
| Isaac-Velocity-Flat-G1 | 4096 | present | present |
| Isaac-Velocity-Flat-H1 | 4096 | present | present |
| Isaac-Velocity-Flat-UnitreeGo2 | 4096 | present | present |
| Isaac-Velocity-Rough-AnymalD | 4096 | present | present |
| Isaac-Velocity-Rough-Cassie | 4096 | present | present |
| Isaac-Velocity-Rough-G1 | 4096 | present | present |
| Isaac-Velocity-Rough-H1 | 4096 | present | present |
| Isaac-Velocity-Rough-UnitreeGo2 | 4096 | present | present |

The eight environments missing from the current baseline are:

- `Isaac-Lift-Cable-Franka`
- `Isaac-Lift-Cable-Franka-Camera`
- `Isaac-Lift-Cloth-Franka`
- `Isaac-Lift-Cloth-Franka-Camera`
- `Isaac-Lift-Soft-Franka`
- `Isaac-Lift-Soft-Franka-Camera`
- `Isaac-Pendulum-MARL-Direct`
- `Isaac-Shadow-Handover`

There are no WARM runtime/training entries for these eight task names on any
queried machine in the August 14–September 9 window. Pendulum MARL needs RL-Games
or SKRL training; it has no RSL-RL registration. The six camera tasks otherwise
present in the current baseline have no eligible exact-August-28 default runs.

[Environment-default CSV](coverage-environment-defaults.csv) provides the same
flags and available alternative counts for every snapshot.

## Complete current backend/renderer gaps

There are **86 missing pairs out of 156** for each metric: 56 have valid paired
training results at other counts; 30 have no usable paired training record at the
default count. Counting the two graph metrics gives 172 missing combinations out
of 312. Multiple profiles account for the 87 records covering 70 pairs.

**P** = Isaac Sim PhysX, **N** = Newton MJWarp, **V** = Newton MJWarp + VBD,
**O** = OV PhysX. Renderers: **I** = Isaac Sim RTX, **W** = Newton Renderer,
**R** = OV RTX. A dash means no pair-level gaps, not complete preset coverage.
Kamino, the Kamino-only Fourbar task, and incompatible P/R and O/I pairs are excluded.

| Environment | Default envs | Current missing pairs (both metrics) |
| --- | ---: | --- |
| Isaac-Ant-Direct | 4096 | O |
| Isaac-Ant | 4096 | O |
| Isaac-Cartpole-Direct | 4096 | O |
| Isaac-Cartpole | 4096 | O |
| Isaac-Cartpole-Camera-Direct | 512 | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Cartpole-Camera | 512 | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Humanoid-Direct | 4096 | O |
| Isaac-Humanoid | 4096 | O |
| Isaac-Lift-Cable-Franka | 8192 | V |
| Isaac-Lift-Cable-Franka-Camera | 128 | V/I, V/W, V/R |
| Isaac-Lift-Cloth-Franka | 2048 | P, V |
| Isaac-Lift-Cloth-Franka-Camera | 128 | P/I, P/W, V/I, V/W, V/R |
| Isaac-Lift-Franka | 4096 | O |
| Isaac-Lift-KukaAllegro | 4096 | O |
| Isaac-Lift-KukaAllegro-Camera | 4096 | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Lift-Soft-Franka | 2048 | P, V |
| Isaac-Lift-Soft-Franka-Camera | 128 | P/I, P/W, V/I, V/W, V/R |
| Isaac-Open-Drawer-Franka-Direct | 4096 | O |
| Isaac-Open-Drawer-Franka | 4096 | O |
| Isaac-Pendulum-MARL-Direct | 4096 | P, N, O |
| Isaac-Reach-Franka | 4096 | O |
| Isaac-Reach-Franka-OSC | 4096 | O |
| Isaac-Reach-UR10 | 4096 | O |
| Isaac-Reorient-Cube-Allegro-Direct | 8192 | — |
| Isaac-Reorient-Cube-Allegro | 8192 | — |
| Isaac-Reorient-Cube-Shadow-Direct | 8192 | — |
| Isaac-Reorient-Cube-Shadow | 8192 | — |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 1225 | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Reorient-Cube-Shadow-Camera | 1225 | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Reorient-Franka | 4096 | O |
| Isaac-Reorient-KukaAllegro | 4096 | O |
| Isaac-Reorient-KukaAllegro-Camera | 4096 | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Shadow-Handover-Direct | 2048 | O |
| Isaac-Shadow-Handover | 2048 | P, N, O |
| Isaac-Velocity-Flat-AnymalD | 4096 | O |
| Isaac-Velocity-Flat-Cassie | 4096 | O |
| Isaac-Velocity-Flat-G1 | 4096 | O |
| Isaac-Velocity-Flat-H1 | 4096 | O |
| Isaac-Velocity-Flat-UnitreeGo2 | 4096 | O |
| Isaac-Velocity-Rough-AnymalD | 4096 | O |
| Isaac-Velocity-Rough-Cassie | 4096 | O |
| Isaac-Velocity-Rough-G1 | 4096 | O |
| Isaac-Velocity-Rough-H1 | 4096 | O |
| Isaac-Velocity-Rough-UnitreeGo2 | 4096 | O |

[Configuration CSV](coverage-configurations.csv) contains the full matrix for
**both displayed snapshots**, the paired metric statuses, source IDs/dates, and
alternative environment counts. [Observed profiles](coverage-observed-profiles.csv)
retains full measured configurations. [Preset selectors](coverage-preset-selectors.csv)
flags explicitly recorded domain selectors. That checklist is not a claim that
all Cartesian combinations of selectors are compatible or measured.

## Data quality and provenance

The [data-issues CSV](coverage-data-issues.csv) audits target-hardware **training**
entries across August 14–September 9. It replaces the previous all-workflow audit.
Issue counts overlap:

- **3,696** entries lack the database physics column; recovered identities
  specify their metadata source, and unresolved identities remain excluded.
- **1,598** camera entries lack the database renderer column; explicit
  per-entry or successful-task selectors may recover them. Unresolved camera
  renderers remain excluded.
- **182** entries have misleading physics group labels and
  **182** have misleading count labels. Measured counts and
  per-run configuration take precedence; group names remain provenance only.

All selected current records report dirty-source metadata and use 10 measured
training iterations. Their commits identify base source, not a reproducible clean
release checkout. FPS variability above 25% is flagged separately for Collection
and Training in `quality_notes`; these are noisy observations, not proven failures.
Camera dimensions and full default domain profiles are not always recorded.
Missing metadata stays empty; no verified camera dimensions are invented from names.

Dates are ingestion dates interpreted as UTC. Original measurement timestamps
lack timezones and are preserved verbatim. Release uses EA 3.0 as a documentation
label; measured source commits are not asserted to be release tags.

The query window includes 63,631 database records on all machines from
`2026-08-14 <= created_at < 2026-09-10`. The selected machine is
`XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`, one RTX PRO 6000 Blackwell Server Edition GPU,
world size 1. Today's incomplete batch is excluded. Only positive paired metrics
from successful WARM training entries with matching measured default counts and
unambiguous task metadata are selected. Current values may use older dates;
history requires exact requested dates.

## Would a fixed count of 4,096 improve coverage?

No. Applying the same hardware, metadata, paired-metric, and date filters to the
queried data reduces coverage. This comparison changes only the required count.

| Snapshot | Count policy | Environments | Backend/renderer pairs | Recorded configurations |
| --- | --- | ---: | ---: | ---: |
| Current | Task defaults | 36 | 70 | 87 |
| Current | Fixed 4,096 | 27 | 52 | 61 |
| August 28 | Task defaults | 30 | 64 | 64 |
| August 28 | Fixed 4,096 | 25 | 50 | 50 |

Both FPS metrics have the same coverage. A fixed 4,096 adds no environments and
loses nine from the current baseline: Cartpole Camera and Camera Direct, all six
Allegro/Shadow cube-reorientation variants, and Shadow Handover Direct. It also
removes all four currently covered OV PhysX configurations, which were measured
at their 8,192 default. Keep task defaults for this dataset.

## Remaining collection work

1. Run missing backend/renderer pairs at the default counts in the configuration
   matrix, particularly OV PhysX at 4,096 and the camera-specific default counts.
2. Schedule the eight entirely absent environments using supported training
   libraries. Each training run should emit both FPS metrics; separate runtime
   runs are not needed to fill these graph modes.
3. Replace the older current camera values with clean release-source runs that
   record resolved physics, renderer, image types, dimensions, camera count,
   presets, measured environment count, and dependency versions.
4. Increase the measurement window and repeat noisy runs. For historical dates
   without matching records, preserve “Not available”; a new run cannot recreate
   an observation from August 14.
