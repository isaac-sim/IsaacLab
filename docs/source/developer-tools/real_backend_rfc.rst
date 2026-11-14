.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _real-backend-rfc:

RFC: A Real Robot Backend
=========================

.. note::

   **Status: proposal with an exploratory prototype.** No hardware support is claimed.
   ``RealCfg`` and ``RealManager`` are implemented experimentally; ``physics=real`` remains a
   proposed task preset, not an available command-line selection.

An accompanying CPU prototype runs an unchanged ``ManagerBasedEnv`` with stock joint actions and
observations against a fake deployment session. It implements only the interfaces that this path
uses, with no IsaacDeploy C++ integration, and reports unsupported callers as errors. Run it with:

.. code-block:: bash

   uv run python -m isaaclab_experimental.real.example --steps 30

The prototype and its Python deployment API findings live in
``source/isaaclab_experimental/isaaclab_experimental/real/README.md``. The architecture below
describes the broader intended integration; the prototype does not implement all of it.

Motivation and scope
--------------------

Add a ``real`` backend so supported Isaac Lab task code can read a physical robot and send commands
through the existing articulation and sensor interfaces. Selecting the backend should replace the
implementation behind the robot, while preserving joint selectors, tensor shapes, observation
processing, and action processing. This enables policy evaluation, teleoperation, and recording
through the same policy-facing interfaces used in simulation.

The initial scope is one robot, one environment, joint position commands, joint position and velocity
measurements, and a task explicitly configured for hardware. Additional command modes and sensors
are capabilities of individual adapters. Hardware support for every registered task, unrestricted
training on hardware, and automatic physical scene resets are outside this proposal's initial scope.

MuJoCo Playground and REAF provide a useful reference: REAF separates task logic from a device layer
that supplies measurements and accepts commands. Playground builds its simulation and hardware
workflows around that boundary. Deployment selects the policy-facing task components and replaces
the devices; it does not make privileged training state available on hardware.
See the `REAF architecture`_ and `Playground deployment example`_.

Existing extension points
-------------------------

Isaac Lab already has most of the public boundary:

* ``Articulation`` and sensors select backend implementations through ``FactoryBase`` using the
  active manager's ``backend_name``. A key can use explicit registration, as the prototype does, or
  map to an optional package such as ``isaaclab_real``. No core backend allowlist change is needed.
* ``BaseArticulation`` defines state access and buffered position, velocity, and effort targets.
  Existing action terms and observation terms operate on these interfaces.
* ``PhysicsManager`` owns initialization, stepping, readiness events, and shutdown. It can serve as
  the initial lifecycle boundary for a hardware session.
* ``LeappDeploymentEnv`` already binds exported policy inputs and outputs to scene entities. Its
  current constructor, reset, and stepping logic create and advance a simulation.

The construction path also assumes a USD stage, scene authoring, Warp-backed data, and a scene data
provider. Adding a selector entry alone would not remove these assumptions. The relevant extension
contracts are documented in :ref:`backend-architecture` and :ref:`add-physics-backend`.

Proposed architecture
---------------------

Use the existing backend dispatch and add an optional ``isaaclab_real`` package. A proposed
``RealCfg`` selects ``RealManager`` and supplies explicit endpoint, robot identity, joint mapping,
control mode, calibration, and timing configuration. A task exposes ``physics=real`` only after its
hardware configuration passes capability validation.

.. code-block:: text

   Shared task, observations, actions, commands, and recorder
                              |
              Existing articulation and sensor interfaces
                              |
               Backend selected by the task configuration
                    /                         \
        PhysX / Newton / OvPhysX        real: hardware implementations
                 |                            |
         Simulated robot state       Deployment session and robot driver
                                              |
                                         Physical robot

``RealManager`` owns the connection and lifecycle of a deployment session. The hardware articulation
implementation exposes telemetry through ``ArticulationData`` and forwards buffered targets through
the session. Sensor implementations expose the existing data interfaces for supported physical
sensors. They share one session where they refer to the same robot; they do not open independent
connections or maintain competing command buffers.

Retain USD initially as robot description and optional visualization metadata. Physical measurements
remain authoritative, and no dynamics solver advances the robot. Existing spawning and cloning
paths need an explicit hardware configuration: one environment, no simulated actuators or sensor
rendering, and no simulation-only scene objects required by the task. This avoids requiring a
USD-free redesign before validating the backend boundary. Running without USD or Warp is a separate
extension, not a property of the initial backend.

State and command compatibility
-------------------------------

Preserve existing public access patterns, including ``robot.data.joint_pos.torch``, joint-name
selection, ``set_joint_position_target()``, and ``write_data_to_sim()``. For the hardware backend,
the last operation submits buffered commands to the deployment session. It retains its existing
method name for compatibility.

The adapter must preserve:

* A leading environment dimension, initially ``num_envs=1``, and the existing ``ProxyArray`` contract.
* Canonical joint and body ordering, radians and SI units, documented frames, and XYZW quaternions.
* Calibration and conversion between the canonical robot interface and the hardware driver's layout.
* Explicit timestamps and validity for measurements. Derived kinematics use the same measured state;
  unavailable floating-base state requires a configured estimator.

Target setters represent commands. Direct state writes such as ``write_joint_position_to_sim()`` and
root-pose writes represent simulation state replacement. Hardware implementations reject unsupported
state writes; they must not reinterpret them as motion commands. A homing routine is a separate,
configured hardware operation.

Keep command-buffer ownership in the existing actuator collection. Each adapter declares which
hardware control modes and gains it accepts. Simulation actuator models cannot implicitly apply a
second PD controller on top of the physical robot's position controller.

IsaacDeploy integration
-----------------------

IsaacDeploy can provide the deployment runtime behind this backend, but policy inference and robot
I/O are separate responsibilities. Its current ``isaac_deploy_core::InferenceController`` accepts
named tensors and provides creation, warmup, activation, advancement, and deactivation. Device reads,
device writes, and scheduling are supplied by its callers. The ROS packages bind it to ROS topics or
``ros2_control`` interfaces.

The inspected public core is C++ and its build depends on ROS components, LibTorch, and Triton. It does
not expose a Python import suitable for directly constructing a Lab backend. See the pinned
`IsaacDeploy controller interface`_ and `IsaacDeploy build dependencies`_. A Python-facing session
adapter is therefore an integration deliverable. It must reuse the existing deployment interfaces
and robot drivers, preserving their lifecycle and command semantics.

There are two execution paths with distinct owners:

* **Python task execution:** Lab's observation and action processing surround a Python policy or
  teleoperation source. The deployment session provides robot I/O. Exporting the policy is optional;
  IsaacDeploy's inference controller does not run a second copy of the policy.
* **Exported policy execution:** a LEAPP bundle owns the exported observation processing, policy, and
  action processing. IsaacDeploy executes that graph. Lab binds raw measurements and final commands
  to its robot interfaces, as the simulation deployment environment already does with LEAPP.
  Lab must not also apply its Python observation and action transformations.

The initial implementation should agree the session boundary with IsaacDeploy before choosing a
native Python binding or a client of its existing ROS runtime. Both must support the same Lab robot
interfaces. An inference-only binding is insufficient to implement the hardware backend.

Timing and lifecycle
--------------------

Keep the observable sequence: buffer targets, submit commands, complete a backend tick, refresh
measurements, and compute observations. On hardware, completing a tick means waiting for the
configured clock and obtaining measurements with a defined freshness bound.

For the first implementation, ``RealManager.step()`` represents one configured base tick and the
existing decimation determines the policy period. The robot driver owns any faster servo loop.
The adapter validates the requested rate against the driver, exposes measured elapsed time, and
does not perform accelerated catch-up steps after a missed deadline. A deployment runtime that owns
policy scheduling needs an explicit mode with one clock owner; Lab must not run another inference
or decimation loop around it.

Construction and passive validation must not activate actuation. Connection, readiness, activation,
and deactivation are distinct lifecycle transitions. Failed initialization releases acquired
resources. Existing readiness and stop events should be reused where their semantics match.

Hardware reset clears policy history, observation history, command state, and episode bookkeeping.
An explicitly configured reset procedure may perform homing or wait for scene preparation. Simulation
reset events, state teleportation, randomization, and automatic reset after a terminal step cannot
be inherited unchanged. ``forward()`` refreshes available state or kinematics without moving the
robot. Shutdown deactivates the controller and closes the session. Before activation, require finite
command and telemetry deadlines and a defined hold, deactivate, or stop response. The adapter must
identify the driver-side watchdog that enforces command expiry even after the Lab process exits;
IsaacDeploy's inference interface alone does not provide this guarantee.

Task compatibility
------------------

A hardware preset selects actor observations, hardware-capable action terms, external command
sources, and an explicit reset procedure. It disables training-only events and removes privileged
critic inputs. Rewards and terminations may be reused when their inputs are measured or estimated;
otherwise the hardware task omits or replaces them explicitly.

Before activation, validate requested measurements, command modes, observation groups, event terms,
controller requirements, and timeout behavior against adapter capabilities. Missing contact forces,
simulated object poses, Jacobians, or dynamics terms must produce a clear unsupported-capability error. Physical
camera data replaces rendered observations only when calibration, format, and timing match the
task's declared contract. No backend should fabricate unavailable state to satisfy an observation.

Compatibility means the same robot API and supported policy-facing task components. It does not
promise that arbitrary existing environment configurations can run unchanged on hardware.

Alternatives and tradeoffs
--------------------------

* **Launch-only integration:** export a LEAPP bundle and launch IsaacDeploy from Lab. This reuses
  deployment infrastructure with less work, but does not expose physical robots through Lab's
  articulation and sensor interfaces or support its Python task loop.
* **Backend integration (proposed):** retain the robot APIs and select hardware implementations with
  ``real``. This matches the desired workflow and uses existing dispatch, while requiring explicit
  hardware lifecycle, capability validation, and scene-construction changes.
* **A new device-layer framework:** extract all task logic behind a new interface modeled on REAF.
  This offers a broader separation from simulation, but changes existing task APIs and expands the
  migration beyond the first hardware integration.

Validation before claiming hardware support
-------------------------------------------

The implementation should first prove the interface with a recorded or fake deployment session:
joint ordering and calibration, command mapping, measurement validity, tick ownership, unsupported
operations, reset behavior, command expiry after client loss, and cleanup after connection or
telemetry failures. These checks should exercise the backend through the public articulation and environment interfaces.

Then run the same supported observation/action configuration in simulation and on a selected robot.
Compare preprocessing and postprocessing on identical recorded inputs, verify that the device
receives the intended commands, and record timing and state-age measurements. Hardware support is
declared for that specific robot, driver, control mode, and task after the physical rollout.
This RFC provides no such validation result.

Decisions needed for implementation
-----------------------------------

* Which robot and maintained driver provide the first position-control example?
* Can IsaacDeploy provide a Python-facing session binding, or should Lab initially connect to its
  ROS runtime? Which component owns robot I/O in each execution path?
* Which existing scene and asset lifecycle hooks need a capability extension to retain USD metadata
  without initializing simulated actuators or rendered sensors?
* Should the existing ``physics=`` selector retain the public ``real`` key, or should a later shared
  runtime selector distinguish physical hardware from physics engines? The initial proposal reuses
  existing dispatch and makes no selector-wide redesign.

.. _REAF architecture: https://reaf-sim.readthedocs.io/en/latest/user_guide/basics.html
.. _Playground deployment example: https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/tasks/manipulation/pick_cube/so101/deploy/env.py
.. _IsaacDeploy controller interface: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy/blob/f3aa6b89b89fb009c9d3a338feb8c549eb00a1b3/isaac_deploy/isaac_deploy_core/inference_controller/inference_controller.hpp
.. _IsaacDeploy build dependencies: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_deploy/blob/f3aa6b89b89fb009c9d3a338feb8c549eb00a1b3/isaac_deploy/isaac_deploy_core/CMakeLists.txt
