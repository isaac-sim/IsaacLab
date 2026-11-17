:orphan:

Controlling conveyors
=====================

Create a controller after authoring the collision geometry and before the first simulation
reset. This example assumes ``sim`` is a simulation context and each environment already
contains a ``Belt`` collision surface and five parcels named ``Parcel0`` through ``Parcel4``:

.. code-block:: python

   from isaaclab.controllers import Conveyor, ConveyorCfg, ConveyorSurfaceCfg

   conveyor = Conveyor(
       ConveyorCfg(
           surface_specs=(ConveyorSurfaceCfg("{ENV_REGEX_NS}/Belt", velocity=0.35),),
           body_pattern=r"/Parcel[0-9]+$",
           body_count_per_env=5,
       ),
       num_envs=scene.num_envs,
       sim=sim,
   )
   sim.reset()
   conveyor.set_speed_scale(0.5)       # Request 0.175 m/s in every environment.
   conveyor.stop(env_ids=[0])         # Stop only environment 0.
   conveyor.start(env_ids=[0])        # Restore its authored speed.
   conveyor.set_speed_scale(-1.0)      # Reverse the authored travel directions.

``speed_scale`` contains the requested multiplier for each environment.
``surface_velocities`` contains the commanded signed centerline speeds in m/s, before the
startup traction ramp. Commands persist across episode and hard resets. Call ``reset(env_ids)``
after restoring bodies to clear stale forces for those environments, and ``close()`` before
destroying the simulation.

For device-side selection, use ``set_speed_scale(..., env_mask=mask)`` and
``reset(env_mask=mask)`` with a full-environment boolean Warp or Torch mask.
Same-device masks and indices reuse native storage without a host round trip.

The Newton MJWarp implementation uses Newton's conveyor force model. Straight and curved
sections retain their authored relative speeds under each environment's multiplier. Independent
speed profiles can use separate controllers with disjoint collision surfaces. Surface geometry
is stationary; the prescribed motion acts through contact forces.
