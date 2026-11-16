:orphan:

Conveyors and surface grippers
==============================

Isaac Lab provides :class:`~isaaclab.controllers.Conveyor` for prescribed belt motion
and :class:`~isaaclab.assets.SurfaceGripper` for suction tools. Their public configurations
select the implementation through the active physics backend. Scene assets own the geometry;
these APIs control its interaction with parcels.

Conveyor commands
-----------------

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

The Newton MJWarp implementation uses Newton's conveyor force model. Straight and curved
sections retain their authored relative speeds under each environment's multiplier. Independent
speed profiles can use separate controllers with disjoint collision surfaces. Surface geometry
is stationary; the prescribed motion acts through contact forces.

Scene-owned surface grippers
----------------------------

Declare a tool in an :class:`~isaaclab.scene.InteractiveSceneCfg`, alongside its robot and
payloads:

.. code-block:: python

   from isaaclab.assets import SurfaceGripperCfg
   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.utils import configclass

   @configclass
   class SortingSceneCfg(InteractiveSceneCfg):
       suction = SurfaceGripperCfg(
           prim_path="{ENV_REGEX_NS}/Robot/.*/panda_hand",
           target_prim_path="{ENV_REGEX_NS}/Parcel.*",
           tool_offset=(0.0, 0.0, 0.105),
           pad_radius=0.012,
           pad_spacing=0.014,
       )

On Newton, ``prim_path`` selects a rigid tool body; local +Z points from the pads toward the
payload. Matching bodies with a single box collision shape are acquisition candidates. Closing
acquires an eligible payload from its physical pose; it does not attach distant objects.
On PhysX, ``prim_path`` selects an authored Isaac Sim surface-gripper prim, and the existing
CPU-only limitation applies. PhysX-specific configuration imports remain supported.

After simulation initialization, access ``scene["suction"]``. Its ``command`` and measured
``state`` are Warp arrays. Commands use +1 to close, -1 to open and 0 to retain the current
command. States use +1 for sealed, 0 for closing and -1 for open. Stage commands with
``set_grippers_command`` or its index/mask variants; ``scene.write_data_to_sim()`` applies them.
The scene handles state updates and selected-environment resets.

Manager-based tasks can use :class:`~isaaclab.envs.mdp.actions.SurfaceGripperBinaryActionCfg`.
Its negative policy action closes the gripper, while a nonnegative action opens it. Tasks can
specialize this action to add release qualification without reimplementing command delivery.
Managed environments close scene grippers automatically; standalone scene users call
``scene.close()`` before destroying the simulation.
