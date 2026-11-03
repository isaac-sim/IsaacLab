# Pollen Robotics MicroDuck task provenance

These contributed tasks integrate the native Isaac Lab port from
[IsaacLab-MicroDuck main `0feb47b15e193c4317289ddada3dfcc33f6a543d`](https://github.com/isaac-sim/IsaacLab-MicroDuck/tree/0feb47b15e193c4317289ddada3dfcc33f6a543d).
The original contributed flat/rough walking recipes remain in `flat_env_cfg.py` and
`rough_env_cfg.py`; the imported walking recipes are distinct `-Pollen` registrations.
Deployment clones in `assets.py` preserve the imported recipes' explicit BAM electrical
ceiling and viscous damping without modifying the original shared walking defaults.

The original task port used Pollen Robotics' Apache-2.0-licensed
[`microduck_rl` at `d424a0c899f6b33cbd3daeb279913134349c0b63`](https://github.com/pollen-robotics/microduck_rl/tree/d424a0c899f6b33cbd3daeb279913134349c0b63).
Historical task comments refer to original/tasks2/tasks3/tasks4 source-extraction section numbers;
these identify the corresponding walking, roller/recovery, seated/ball/trick, and roller-slope task source
snapshots at that pin. Those scratch extractions are not runtime or installed-package inputs.
Current SitStand hardening, VelStand and full-contact site contracts follow
[`bae39f5900b318f7e87fc485c30a31213cfa14eb`](https://github.com/pollen-robotics/microduck_rl/tree/bae39f5900b318f7e87fc485c30a31213cfa14eb).
The adapted SitStand notice and full Apache license are in `sitstand/UPSTREAM_NOTICE.md`;
the VelStand derivation and Apache license are in `velstand/agents/LICENSE-microduck_rl.md`.
The original port authorship is retained in Isaac Lab's `antoiner/feat/microduck` history.

All eight prebuilt robot USDs load through `ISAACLAB_NUCLEUS_DIR`. There is no MJCF conversion,
local asset fallback, or teacher checkpoint in this package. Model geometry remains copyright
Pollen Robotics. The original upstream README distinguishes its Creative Commons BY-SA-NC
model license from Apache-2.0 code; the
[pinned model-license notice](https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/LICENSE-MODELS.md)
clarifies CC BY-NC-SA 4.0 for STL geometry and Apache-2.0 for MJCF/XML and export metadata.
Code licensing does not relicense hosted model geometry.

The two full-contact variants preserve the port's `max_current=0` and electrical joint effort
ceiling of approximately 1.06755316 Nm. This is a local deviation: the pinned Rhoban BAM XL330
constructor at `62bd8ce12154340be97e06f7f41a0ca8f116d967` defaults to 1.75 A. The walking,
curated-contact and roller variants retain that 1.75 A limiter. Full-contact backlash adds
fourteen collider-free 1 mg rotor proxies; original physical link inertias remain preserved.

Teacher weights are fetched into a private cache by the VelStand runner and verified by
SHA256. Their training repository does not declare a weight license; weights are not bundled
or redistributed by this port. A completed training job is not evidence of policy quality:
the imported final StandUp checkpoints have a known back-recovery failure under the final
reset distribution. This integration does not repair or retrain those policies.
