# MicroDuck attribution

The task configurations and MDP terms derive from Pollen Robotics'
Apache-2.0-licensed [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl):

| Components | Source revision |
| --- | --- |
| Walking, recovery, seated, ball and roller tasks | [`d424a0c`](https://github.com/pollen-robotics/microduck_rl/tree/d424a0c899f6b33cbd3daeb279913134349c0b63) |
| SitStand perturbations, VelStand and full-contact site definitions | [`bae39f5`](https://github.com/pollen-robotics/microduck_rl/tree/bae39f5900b318f7e87fc485c30a31213cfa14eb) |

The adapted-code notices and Apache license are retained in
[`sitstand/UPSTREAM_NOTICE.md`](sitstand/UPSTREAM_NOTICE.md) and
[`velstand/agents/LICENSE-microduck_rl.md`](velstand/agents/LICENSE-microduck_rl.md).

Robot geometry is copyright Pollen Robotics. Under the
[model-license notice](https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/LICENSE-MODELS.md),
STL geometry and converted mesh adaptations use CC BY-NC-SA 4.0; MJCF/XML and export metadata
use Apache-2.0. Code licensing does not replace the model license.

All eight robot USDs load through `ISAACLAB_NUCLEUS_DIR`. Full-contact backlash uses fourteen
collider-free 1 mg rotor proxies while retaining the physical links' masses and inertias.
Full-contact servos use `max_current=0` and an electrical joint effort ceiling of approximately
1.06755316 Nm. Walking, curated-contact and roller servos use a 1.75 A firmware current limit.

VelStand downloads its teacher weights into a private cache and verifies their SHA256 hashes.
The training repository does not declare a weight license; weights are not bundled with this package.
