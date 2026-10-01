# Attribution for BAM reference samples

`bam_xl330_m6_goldens.npz` contains motor/friction samples and fitted coefficients from
[BAM (Better Actuator Models)](https://github.com/Rhoban/bam) by Marc Duclusaud and
Grégoire Passault. The upstream Apache-2.0 license is included in `LICENSE-BAM`.

Source revision: `62bd8ce12154340be97e06f7f41a0ca8f116d967` (`mjlab_frictionloss`).
The fit comes from `bam/params/xl330/m6.json`; firmware constants come from
`XL330Actuator` in `bam/dynamixel/actuator.py`. Samples use `kp = 200` and `vin = 7.4`.

Tests construct `BamMotorCfg` from the recorded coefficients (`R` becomes `resistance`).
The metadata also retains `q_offset` and `armature` for provenance; neither is a drive parameter.
Regenerate the fixture with `scripts/tools/generate_bam_goldens.py`.
