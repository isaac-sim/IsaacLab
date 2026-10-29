# Isaac Lab Newton Simulation interfaces

With the MuJoCo Warp solver, `RigidBodyMaterialCfg(friction_combine_mode="multiply")`
opts a material into sliding-friction multiplication. For equal-priority geoms,
the contact coefficient is `mu_a * mu_b`, with MuJoCo's usual `mjMINMU` floor.
This works with both `MJWarpSolverCfg.use_mujoco_contacts` settings and with
runtime material randomization. The ground coefficient already in the model remains
an independent material value: a ground coefficient of 1 preserves the other
material's value; 0.5 scales it by 0.5.

Unauthored modes retain native MuJoCo combination. When modes differ, an explicit
`max` overrides `multiply`; `multiply` overrides `min` and `average`. This extension
implements only multiplication: contacts without a winning `multiply` retain
native behavior. Unequal geom priorities and native MuJoCo explicit contact pairs
still take precedence. Torsional/rolling friction and other contact parameters
are unchanged. Heightfield terrain conversion forwards only the combine mode;
it retains the existing builder defaults for friction and other solver parameters.
It does not add support for importing bound friction coefficients on that path.

This is a Warp contact extension, not full PhysX material compatibility: Newton
still uses one sliding coefficient, rather than separate static/dynamic values.
`use_mujoco_cpu=True` rejects the opt-in, and a standalone exported MJCF does not
encode this callback. The extension does not change the selected friction cone
or raise MuJoCo's native friction floor.
