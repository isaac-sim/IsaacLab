.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _prepare-asset-for-newton:

Prepare an Asset for Newton with MJWarp
=======================================

.. seealso::

   This page is the source of truth for the ``isaaclab-preparing-assets-for-newton`` agent skill
   (`skill source
   <https://github.com/isaac-sim/IsaacLab/blob/develop/skills/user/prepare-assets-for-newton/SKILL.md>`__).
   When you change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

Prerequisites
-------------

Understand how Isaac Lab selects a backend and its task-specific preset before changing an asset;
see :ref:`backends-and-presets`. This guide prepares an asset and task for
``physics=newton_mjwarp``. After both backends run the asset and task, use
:doc:`/source/how-to/transfer_policies_between_physx_and_newton` to transfer a policy checkpoint.
For the conceptual differences that require target-solver validation, see
:ref:`solver-differences`; use :doc:`/source/how-to/solver_tuning/tune_mjwarp` for the focused MJWarp tuning
procedure.

Import a multi-physics asset
----------------------------

Isaac Lab assets can work with both ``physics=physx`` and ``physics=newton_mjwarp``. Many assets
author USD Physics and PhysX schemas; Newton parses the supported authored properties into its
model, so a separate Newton-only copy is not required. Confirm each property against the supported
feature and schema documentation: an authored PhysX attribute can be present without MJWarp using
it during stepping.

For a new URDF or MJCF asset, use the Isaac Lab importers to create a multi-physics asset. Keep
``run_asset_transformer`` and ``run_multi_physics_conversion`` enabled (both default to ``True``)
so the conversion creates neutral physics, PhysX, and MuJoCo payloads. The current converter uses a
nested rigid-body structure rather than the earlier flat USD structure. See
:doc:`/source/how-to/import_new_asset`, :class:`~isaaclab.sim.converters.UrdfConverterCfg`, and
:class:`~isaaclab.sim.converters.MjcfConverterCfg`.

Separate common and solver-specific properties
-----------------------------------------------

Put common USD Physics properties in solver-common configuration classes such as
``RigidBodyBaseCfg`` and ``JointDriveBaseCfg``. Put backend-only properties in the matching
subclasses:

* Use ``MujocoRigidBodyPropertiesCfg``, ``MujocoJointDrivePropertiesCfg``, and
  ``MujocoCollisionCfg`` for MJWarp-specific properties.
* Use the matching ``Newton*PropertiesCfg`` classes for supported Newton-native collision,
  material, articulation, and related properties.
* Keep PhysX-only damping, stabilization, solver-iteration, friction-patch, and compliant-contact
  properties in the matching ``Physx*PropertiesCfg`` classes.

For configuration hierarchy and parameter-to-USD routing, see
:doc:`/source/overview/core-concepts/schema_cfgs`. Check the
:doc:`Newton/MuJoCo schema API </source/api/lab_newton/isaaclab_newton.sim.schemas>` and
:doc:`PhysX schema API </source/api/lab_physx/isaaclab_physx.sim.schemas>` for supported fields.
A value present in a PhysX asset or imported Newton model is not proof that MJWarp consumes it.

Audit the mechanical model
--------------------------

Inspect every dynamic link and contact-relevant object:

* Author positive mass [kg], center of mass [m], and positive-definite inertia [kg*m^2]. Treat a
  placeholder inertia warning as a modeling failure, and verify the inertia and center-of-mass
  frames.
* Apply ``UsdPhysics.CollisionAPI`` only to intentional collision geometry. Check approximation,
  mesh scale, contact offset or margin, material binding, restitution, and self-collision filters;
  visual parity is not collision parity.
* Verify the articulation root, fixed-base representation, fixed-joint merging, nested rigid
  bodies, joint types, axes, and limits.
* Make each body-level gravity setting intentional. A stale override can make a task differ from
  scene gravity.

Match collision, contact, and friction behavior
-----------------------------------------------

Do not copy PhysX friction settings numerically. First verify intended collision shapes, material
bindings, contact locations, contact count, and available gripper normal force in both backends.
Then inspect the resolved per-shape MuJoCo contact dimensionality, ``condim``: ``1`` is
frictionless, ``3`` adds tangential friction, ``4`` adds torsional friction, and ``6`` also adds
rolling friction. Choose the smallest model that represents the contact physics and confirm that
the importer and contact path preserve it.

Tune material friction against measured tangential slip only after the contact model is valid.
``MujocoCollisionCfg`` also exposes expert per-collider overrides such as ``priority``, ``solmix``,
``solref``, and ``solimp``; use them only with a measured contact-model need. Use
:attr:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg.contact_margin`,
:attr:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg.contact_gap`, and
:attr:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionCfg.max_hull_vertices` instead of raw
importer attributes. Follow :ref:`mjwarp-solver-tuning` for the current global solver and contact
tuning sequence.

.. _newton-velocity-limits:

Validate actuators and limits
-----------------------------

Audit per-joint effort limits, stiffness, damping, friction, armature, action scale, and control
period. Armature applies only to articulated coordinates: use a physically justified reflected
motor/transmission inertia or a controlled response test, and do not use it to hide bad body mass,
inertia, units, reset penetration, or contact capacity. Retune damping after changing armature.
For the general actuator model and configuration guidance, see :ref:`overview-actuators`.

``actuator_velocity_limit`` is a physical rated speed that Isaac Lab can use in task or actuator
logic. ``joint_velocity_limit`` requests a solver-side clamp. Isaac Lab writes the latter to
Newton's model, but MJWarp drops it when constructing its solver model; Kamino honors it. MJWarp
does not enforce either value while stepping. When a task needs a speed bound under MJWarp,
implement and validate it in task or control logic. Use ``joint_effort_limit`` for solver limits,
``actuator_effort_limit`` to clip explicit actuator models, and physically justified controller
behavior to keep the response well behaved.

Run paired smoke tests
----------------------

Run the same fixed task state through both backends before training or transfer:

.. code-block:: bash

   uv run --extra isaacsim python scripts/environments/zero_agent.py --task TASK --num_envs 4 --viz none physics=physx
   uv run --extra isaacsim python scripts/environments/zero_agent.py --task TASK --num_envs 4 --viz none physics=newton_mjwarp

Let each run cross multiple resets. Record object displacement, contact count, gripper effort,
penetration, and success rate for the same fixed grasp. Also check for non-finite state, first-step
impulses, unexpected saturation, excessive angular velocity, contact loss, and importer or solver
warnings. Reject robot-object and robot-support penetration, impossible mimic states, and invalid
randomized geometry before the first physics step.

Account for solver differences
------------------------------

After the paired smoke tests, use the target solver's controls to address the
differences described in :ref:`solver-differences`:

#. Revalidate contact behavior with the smallest useful environment count and a
   visualizer before scaling up.
#. Retune material friction from measured slip; PhysX patch friction and
   MJWarp contact friction are not numerically interchangeable.
#. Retune restitution from observed bounce and chatter rather than assuming a
   PhysX scene threshold applies to Newton.
#. Compare timestep and ``num_substeps`` against the fixed reproduction,
   especially for contact-heavy tasks.
#. When a PhysX task used CCD, validate a Newton collision strategy and shorter
   solver timestep as needed; MJWarp's ``ccd_iterations`` is not a CCD switch.
#. For Kamino, validate reset-state consistency and only then investigate
   constraint stabilization or convergence settings.

Diagnose Newton-only failures
-----------------------------

Reproduce the first failing step with one environment, a fixed seed and reset state, no domain
randomization, and the same action sequence in both backends. Localize the failure before tuning:

#. At initialization or the first step, inspect mass, inertia, scale, reset overlap, topology,
   drives, and unsupported features.
#. At contact onset, inspect contact locations and counts, capacity warnings, margins, ``condim``,
   friction, cone choice, and extreme mass or inertia ratios.
#. Under control, inspect effort and gain limits, action scale, ``dt``, substeps, damping,
   armature, and joint-limit impacts.
#. In dense scenes, compare the busiest environment with per-environment contact and constraint
   capacities.

Enable ``NewtonCfg.debug_mode`` to inspect iteration-cap usage. Increase overflowing capacity first;
change convergence settings only after the asset, reset, controller, contact model, and capacities
are valid. Keep the smallest fixed-state reproduction and record the first non-finite quantity so
later changes can be compared one at a time.
