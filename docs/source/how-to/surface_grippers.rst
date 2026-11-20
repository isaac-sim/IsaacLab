:orphan:

Scene-owned surface grippers
============================

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
The four-pad Newton model uses a fixed 4 N force limit per pad, a 6 mm acquisition gap
and a 25 mm break distance. Tool offset defaults to the body origin; pad spacing and pad radius
describe its hardware.
On PhysX, ``prim_path`` selects an authored Isaac Sim surface-gripper prim, and the existing
CPU-only limitation applies. PhysX-specific configuration imports remain supported;
nondefault Newton target and pad settings are rejected by PhysX.

After simulation initialization, access ``scene["suction"]``. Its ``command`` and measured
``state`` are Warp arrays. Commands use +1 to close, -1 to open and 0 to retain the enabled
vacuum state. The staged command remains zero. States use +1 for sealed, 0 for closing and
-1 for open. Stage commands with ``set_grippers_command`` or its index/mask variants;
``scene.write_data_to_sim()`` applies them.
The scene handles state updates and selected-environment resets.

Manager-based tasks can use :class:`~isaaclab.envs.mdp.actions.SurfaceGripperBinaryActionCfg`.
Its negative policy action closes the gripper, while a nonnegative action opens it. Tasks can
specialize this action to add release qualification without reimplementing command delivery.
Managed environments close scene grippers automatically; standalone scene users call
``scene.close()`` before destroying the simulation.
Custom ``SurfaceGripperCfg.class_type`` constructors continue to receive only the
configuration argument; the standard Lab factory receives the scene's simulation context.
