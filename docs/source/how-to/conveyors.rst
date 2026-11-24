:orphan:

Controlling conveyors
=====================

.. _tutorial-conveyor-suction:

Using a conveyor with suction
-----------------------------

This standalone Newton MJWarp example has one static belt, one parcel, and one kinematic
suction tool. It builds the scene from primitives. The belt moves the parcel, the tool seals
against its top face and lifts it, and opening the gripper releases it onto the belt.

.. code-block:: bash

   uv run python scripts/tutorials/01_assets/run_conveyor_suction.py --device cuda:0 --viz newton

Use ``--viz none`` to run the same sequence without a window.

Create the scene and conveyor before resetting the simulation. Name the belt's collision
surface and the parcel body in the controller configuration. The belt and parcel use low
contact friction because the conveyor controller supplies their tangential traction.
After stopping the belt, command the kinematic tool above the parcel with its pads facing
down. Stage the gripper command, write the scene before stepping, and refresh it afterward.
Close the conveyor and scene before tearing down the simulation.

.. dropdown:: Code for run_conveyor_suction.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/01_assets/run_conveyor_suction.py
      :language: python
      :linenos:

See :ref:`tutorial-interact-surface-gripper` for backend-specific gripper configuration.

Configuring a controller
------------------------

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
   conveyor.set_speed_scale(0.5)
   conveyor.stop(env_ids=[0])
   conveyor.start(env_ids=[0])
   conveyor.set_speed_scale(-1.0)

``speed_scale`` contains the requested multiplier for each environment. It multiplies
the configured signed centerline velocity before the startup ramp. Commands persist across
episode and hard resets. Call ``reset(env_ids)``
after restoring bodies to clear stale forces for those environments, and ``close()`` before
destroying the simulation.

Removing solver hooks invalidates Newton's cached step graphs. Newton rebuilds them on a
subsequent step when capture is enabled. If you capture the simulation in an enclosing CUDA
graph, discard that graph before changing the hooks and capture again before replaying.

For device-side selection, use ``set_speed_scale(..., env_mask=mask)`` and
``reset(env_mask=mask)`` with a full-environment boolean Warp or Torch mask.
Same-device masks and contiguous int32 indices reuse native storage without a host round trip.
Strided or int64 indices require normalization. Surface paths follow the simulation's clone
template, or ``/World/envs/env_{}`` for manually built worlds without a clone plan.

The Newton MJWarp implementation uses the force model from the conveyor example in the locked
Newton revision. A public runtime interface is tracked in
`Newton #5501 <https://github.com/newton-physics/newton/issues/5501>`__.
Straight and curved sections retain their authored relative speeds under each environment's
multiplier. Independent speed profiles can use separate controllers with disjoint collision surfaces. Newton requires
world-attached static collision shapes for the belt surfaces. Contact forces produce the
prescribed surface motion.
