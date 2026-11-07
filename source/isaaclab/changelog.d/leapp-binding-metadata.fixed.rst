* Labeled joint, body, gain, and command instance axes as LEAPP systems. Exported planar
  velocity commands as built-in twist setpoints with ``vx``, ``vy``, and ``wz`` components,
  and replaced custom world-frame ray-caster pose kinds with built-in quantities.
* Standardized linear/angular velocity and twist component labels across asset and sensor
  annotations, and used LEAPP quantity helpers to validate raw built-in inputs and PD gains.
