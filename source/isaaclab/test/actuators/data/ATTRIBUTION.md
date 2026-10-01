# Attribution for vendored BAM actuator parameters

The coefficients in `bam_xl330_m6.usda` and the reference samples in
`bam_xl330_m6_goldens.npz` are derived from the
[BAM (Better Actuator Models)](https://github.com/Rhoban/bam) project by
Marc Duclusaud and Grégoire Passault, licensed under the Apache License 2.0.
The upstream license is included in `LICENSE-BAM`.

- Upstream repository: <https://github.com/Rhoban/bam>
- Branch: `mjlab_frictionloss`
- Commit: `62bd8ce12154340be97e06f7f41a0ca8f116d967`

## Provenance of each field

| Fields | Upstream source |
| --- | --- |
| `actuator`, `model`, `kt`, `R`, `armature`, `q_offset`, `friction_*`, `dtheta_stribeck`, `alpha`, `load_friction_*` | `bam/params/xl330/m6.json` (copied verbatim; identification result for the Dynamixel XL330 with the `m6` friction model) |
| `error_gain`, `max_pwm`, `max_current`, `kp`, `vin` | `bam/dynamixel/actuator.py`, `XL330Actuator.__init__` (firmware/supply constants, which upstream keeps in code rather than in the parameter file). `error_gain` is the evaluated form of `(4096 / (2 * pi)) / (256 * 885)`. |

The USD fixture stores `R` as `resistance` and `kp` as `kpFw`. Its firmware gain and
supply voltage use the upstream XL330 defaults. They are nominal values only: the
per-environment firmware gain and supply voltage are configured through
`BamActuatorCfg` and consumed by the native Newton drive (the reference
goldens in `bam_xl330_m6_goldens.npz`, for instance,
use `kp = 200` and `vin = 7.4`).

`q_offset` is the calibration offset of the identification testbench. It is kept in the
golden metadata for provenance and is not used by the Isaac Lab actuator model.
`armature` is also retained in that metadata; solver inertia is owned by the joint.
