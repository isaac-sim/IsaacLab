.. _overview-actuators:


Actuators
=========

An articulated system is driven through its actuated joints (its degrees of freedom). On a physical
robot the joints are moved by active components -- electric or hydraulic motors -- or resisted by
passive ones such as springs and friction. These components introduce non-linear characteristics:
finite torque, bounded speed, transmission delays, and gearbox effects.

Isaac Lab exposes two ways to reproduce that behavior in simulation:

* **Implicit actuators** hand the position/velocity gains to the physics engine, which runs a
  spring-damper (PD) controller in its discrete solver. This is a low-overhead default when the
  built-in PD model is sufficient. PhysX implicit drives may tolerate higher gains or a larger time
  step in isolation, but contacts, mimic joints, joint limits, large target jumps, and insufficient
  solver convergence can still destabilize them. See the `PhysX articulation drive stability
  guidance
  <https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Articulations.html#articulation-drive-stability>`_.
* **Explicit actuators** normally run an Isaac Lab model every step to compute a joint torque, clip
  it to the motor's capabilities, and submit only the resulting effort. Supported native actuator
  paths bypass the collection's processed-command buffers: Newton finishes processing inside the
  solver, while the PhysX adapter processes commands during ``write_data_to_sim()``. Explicit
  models trade some cost for the ability to model saturation, delay, gearing, or a learned drive.

Every actuator group -- implicit or explicit -- is configured on
:attr:`~isaaclab.assets.ArticulationCfg.actuators` and exposed by the runtime
:class:`~isaaclab.actuators.ActuatorCollection` on
:attr:`~isaaclab.assets.Articulation.actuators`. The collection routes groups and stages commands
and telemetry. Ordinary runtime code sets desired targets through the collection; the articulation
owns actuator execution and command submission. LEAPP-exported action terms are the temporary
exception described in :ref:`actuators-migrating-setters`.

.. contents:: On this page
    :local:
    :depth: 1


Quick usage
-----------

Declare one or more actuator groups on the articulation config. Each group matches a disjoint set
of joints by regular expression and picks a model:

.. code-block:: python

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    robot_cfg = ArticulationCfg(
        spawn=...,  # your USD / spawner config
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*"],
                stiffness=40.0,
                damping=2.0,
                joint_effort_limit=80.0,
            ),
        },
    )

At runtime, send commands through :attr:`~isaaclab.actuators.ActuatorCollection.command`. Position
and velocity commands are expressed in joint-side coordinates, and every command buffer is indexed
by articulation joint. The setters are keyword-only and default to all environments and all joints:

.. code-block:: python

    import torch

    # desired position for every joint of every environment
    values = torch.full((robot.num_instances, robot.num_joints), 0.5, device=robot.device)
    robot.actuators.command.set_position_index(value=values)

You never call the models yourself: the articulation stages and submits actuator commands inside
:meth:`~isaaclab.assets.Articulation.write_data_to_sim`. Isaac Lab-managed models and the PhysX
native adapter run during this call; Newton-native controllers finish processing inside the solver.
The rest of this page explains what happens between the desired command you set and the command
consumed by physics, then documents every gain and limit with side-by-side comparisons.


.. _actuators-pipeline:

The actuator pipeline
---------------------

Setting an actuator command does not talk to the solver directly. The diagram shows three execution
paths after collection staging:

#. **Actuator command** -- ``actuators.command.set_*_index`` / ``_mask`` write the desired
   position, velocity, and effort into buffers expressed in joint-side coordinates.
#. **ActuatorCollection** -- routes groups and stages full-articulation commands, processed joint
   commands, and telemetry. It does not own actuator-model gains or scratch state.
#. **Isaac Lab actuator** -- a Lab-managed explicit model turns the actuator command into a joint
   effort and clips it before submission.
#. **Implicit drive** -- the backend consumes the desired targets and the solver-side PD gains.
#. **Newton actuator** -- a native explicit controller follows the active backend path; the next paragraph gives its stage.

For Isaac Lab-managed models, ``actuators.joint_command`` exposes the processed position, velocity,
and effort commands submitted to the active physics backend. Native paths bypass this view, so it
is not submitted-command telemetry for them. The PhysX native adapter processes commands during
``write_data_to_sim()``; Newton-native controllers finish inside the solver. For supported
Newton-native models and their limitations, see :ref:`actuators-native`.

.. figure:: ../../_static/actuators/pipeline-light.png
    :class: only-light
    :align: center
    :width: 90%
    :alt: Three actuator paths: Lab-managed models execute before submission, implicit drives
        execute in the solver, and native models execute inside the Newton/MJWarp solver or in the
        PhysX adapter before submission.

.. figure:: ../../_static/actuators/pipeline-dark.png
    :class: only-dark
    :align: center
    :width: 90%
    :alt: Three actuator paths: Lab-managed models execute before submission, implicit drives
        execute in the solver, and native models execute inside the Newton/MJWarp solver or in the
        PhysX adapter before submission.

Where the gains land differs by path, and this is the single most common source of confusion:

* For an **implicit** group, :attr:`~isaaclab.actuators.ActuatorBaseCfg.stiffness` and
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.damping` are written straight into the solver, which
  runs the PD law. ``compute()`` passes the desired targets through unchanged while recording an
  approximate torque for telemetry.
* For an **explicit** group, the same gains are consumed by the model to compute a torque, and the
  solver's own PD gains for those joints are set to zero. Reading ``data.joint_stiffness`` or
  ``data.joint_damping`` on an explicit joint therefore returns **zero** -- the gains live in the
  actuator model, not the solver.

The actuator gains configured on a named group are distinct from
:attr:`~isaaclab.assets.ArticulationData.joint_stiffness` and
:attr:`~isaaclab.assets.ArticulationData.joint_damping`. The latter report the solver drive gains:
they match implicit actuator gains after those gains are written to the solver, and are zero for
explicit actuator joints. They are not a collection-wide mirror of the actuator-model gains.


.. _actuators-joint-property-ownership:

Joint and actuator property ownership
--------------------------------------

An actuator configuration uses its joint expression as a convenient way to select joints, but its
joint-property fields are construction-time overrides. Isaac Lab resolves and writes
``joint_effort_limit``, ``joint_velocity_limit``, armature, friction, and implicit drive gains when
the articulation is built. Their live runtime state belongs to
:class:`~isaaclab.assets.ArticulationData`, including ``joint_effort_limits``, ``joint_vel_limits``,
``joint_armature``, and the supported friction properties. Read or change those values through the
articulation data and joint writers; :class:`~isaaclab.actuators.ActuatorCollection` deliberately
has no joint-property API.

The deprecated group properties ``effort_limit_sim``, ``velocity_limit_sim``, ``armature``,
``friction``, ``dynamic_friction``, and ``viscous_friction`` remain forwarding views through 3.x.
Assignment emits a warning and forwards the group selection and value through
the corresponding articulation joint writer. It does not recreate an actuator
or collection-level property buffer. See
:ref:`actuators-solver-limit-migration` for the replacement data views and writers.

Ordinary explicit actuators retain only model state, such as their ``effort_limit``, rated
``velocity_limit``, gains, delay, and motor curve. They do not own solver-limit or friction copies.
Implicit actuators are different because the backend executes their drives: their stiffness,
damping, and effort projection read the live articulation properties. Native backend controllers
similarly own their controller gains; those gains are not joint solver gains.

``effort_limit`` and ``velocity_limit`` describe an actuator model. ``joint_effort_limit`` and
``joint_velocity_limit`` describe a joint/solver request and may have different values. In
particular, an explicit actuator's ``effort_limit`` clips its model output, while
``joint_effort_limit`` constrains the solver. ``velocity_limit`` is the model's rated joint-side
speed or implicit soft-limit snapshot; ``joint_velocity_limit`` requests a solver constraint.
Solver velocity enforcement is backend-dependent, so it is not a portable clamp. For implicit
actuators, bare ``effort_limit`` remains only as a deprecated alias for ``joint_effort_limit``.

.. note::

    Each call to :meth:`~isaaclab.assets.Articulation.write_data_to_sim` runs collection staging and
    submission; Newton-native processing then continues inside the solver. Call it before advancing
    the simulation, or rely on the scene or environment loop to call it. Telemetry buffers
    (:attr:`~isaaclab.actuators.ActuatorCollection.computed_torque`,
    :attr:`~isaaclab.actuators.ActuatorCollection.applied_torque`) reflect the most recent call.


Choosing a model
-----------------

The models share the configuration base (:class:`~isaaclab.actuators.ActuatorBaseCfg`). The PD
models differ in their clipping and state; neural models replace the PD model with a learned torque
predictor. Pick the simplest model that captures the effect you need.

.. list-table::
    :header-rows: 1
    :widths: 22 34 24 20

    * - Model (config)
      - Torque / clipping
      - Where limits clip
      - Extra config fields
    * - :class:`~isaaclab.actuators.ImplicitActuator`
        (:class:`~isaaclab.actuators.ImplicitActuatorCfg`)
      - Solver runs the PD law from the written gains.
      - ``joint_effort_limit`` clips in the solver.
      - --
    * - :class:`~isaaclab.actuators.IdealPDActuator`
        (:class:`~isaaclab.actuators.IdealPDActuatorCfg`)
      - :math:`\tau = k_p (q_{des}-q) + k_d(\dot{q}_{des}-\dot{q}) + \tau_{ff}`
      - Model clips directly to :math:`\pm\,\tau_{max}` (``effort_limit``).
      - --
    * - :class:`~isaaclab.actuators.DCMotor`
        (:class:`~isaaclab.actuators.DCMotorCfg`)
      - Same PD torque, clipped to a four-quadrant torque-speed envelope.
      - Model clips against a velocity-dependent limit.
      - ``saturation_effort``, ``velocity_limit``
    * - :class:`~isaaclab.actuators.DelayedPDActuator`
        (:class:`~isaaclab.actuators.DelayedPDActuatorCfg`)
      - Ideal PD applied to commands delayed by a circular buffer.
      - Same as ideal PD (``effort_limit``).
      - ``min_delay``, ``max_delay``
    * - :class:`~isaaclab.actuators.RemotizedPDActuator`
        (:class:`~isaaclab.actuators.RemotizedPDActuatorCfg`)
      - Delayed PD with an angle-dependent torque ceiling.
      - Torque clipped by a joint-angle lookup table.
      - ``joint_parameter_lookup``
    * - :class:`~isaaclab.actuators.ActuatorNetMLP` /
        :class:`~isaaclab.actuators.ActuatorNetLSTM`
      - A trained network predicts the torque from the joint history.
      - Network output clipped by the DC-motor envelope.
      - ``network_file`` (+ input scaling)

**ImplicitActuator.** The default. Gains and solver limits are handed to the solver. The collection
also estimates effort telemetry from the current state because the solver does not expose that value
on every backend.

**IdealPDActuator.** The reference explicit model: a PD controller with feed-forward effort and a
symmetric joint-side torque clip at :math:`\pm\,\tau_{max}`. Use it when you want explicit-actuator
semantics (a hard effort ceiling enforced in the model) without a specific motor curve.

**DCMotor.** Extends the ideal PD with a linear four-quadrant DC-motor torque-speed curve: the
achievable torque falls off as the joint spins faster, so a motor cannot produce peak torque at peak
speed. Requires ``saturation_effort`` (the stall torque) and ``velocity_limit`` (the no-load speed).

**DelayedPDActuator.** An ideal PD whose position, velocity, and effort commands pass through a
circular delay buffer. At each reset the lag is drawn uniformly in ``[min_delay, max_delay]`` physics
steps, so domain randomization over the two bounds produces a spread of transport delays.

**RemotizedPDActuator.** A delayed PD whose torque ceiling depends on the joint angle, interpolated
from ``joint_parameter_lookup`` (a table of angle, transmission ratio, and maximum torque). Use it
for linkages whose effective lever arm changes over the range of motion.

**ActuatorNetMLP / ActuatorNetLSTM.** Learned drives that replace the analytical torque with a
network prediction from the joint-position error and velocity history; the output is clipped by the
DC-motor envelope. They require a trained TorchScript checkpoint and are out of scope for this page
-- see the :mod:`isaaclab.actuators` API reference.


.. _actuators-parameter-reference:

Parameter reference
-------------------

Each subsection below isolates one parameter on a five-way pendulum comparison and shows the
resulting response. All clips come from the same demo, a single-joint pendulum stepped at
:math:`dt = 1/360\text{ s}` with commands issued at 60 Hz; the stiffness, damping, and armature
sweeps run on the *implicit* path.

.. important::

    **Explicit solver effort default.** Explicit groups default
    :attr:`~isaaclab.actuators.ActuatorBaseCfg.joint_effort_limit` to ``1.0e9`` so their model
    output is not clipped by the solver a second time. See
    :ref:`actuators-joint-property-ownership` for model-limit, joint-limit, and implicit-alias
    semantics.


Stiffness
^^^^^^^^^

Stiffness (:math:`k_p`, the proportional gain) sets how hard the controller pulls the joint toward
its position target. Higher stiffness tracks a step faster but overshoots more and, past a point,
excites oscillation; too little stiffness leaves a steady-state error under load. Tune it together
with damping. Units are [N·m/rad] for revolute joints and [N/m] for prismatic joints.

.. figure:: ../../_static/actuators/stiffness-clip.webp
    :align: center
    :width: 100%
    :alt: Five pendulums with increasing stiffness stepping to the same target.

.. figure:: ../../_static/actuators/stiffness-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Position step response for a stiffness sweep.

.. figure:: ../../_static/actuators/stiffness-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Position step response for a stiffness sweep.


Damping
^^^^^^^

Damping (:math:`k_d`, the derivative gain) resists joint velocity and removes energy from the
response. With too little damping a stiff joint rings; increasing it suppresses overshoot until the
joint is critically damped, and beyond that the response turns sluggish (overdamped). Damping is
also how you set a velocity target's tracking gain. Units are [N·m·s/rad] (revolute) or [N·s/m]
(prismatic).

.. figure:: ../../_static/actuators/damping-clip.webp
    :align: center
    :width: 100%
    :alt: Five pendulums from underdamped to overdamped stepping to the same target.

.. figure:: ../../_static/actuators/damping-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Position step response for a damping sweep.

.. figure:: ../../_static/actuators/damping-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Position step response for a damping sweep.


Armature
^^^^^^^^

Armature [kg or kg·m², depending on joint type] models the reflected rotor inertia of the
drivetrain: it is added directly to the joint-space inertia. Physically it captures the gearbox and
motor inertia a real drive carries. Increasing a physically justified armature lowers the drive's
effective natural frequency and can improve numerical conditioning, especially for low-inertia
joints, but it also changes the physical model. In this single-joint comparison, more armature
makes the response more sluggish and increases the margin before the chosen gains become unstable.

Both paths run at discrete solver steps, but explicit models submit an effort while implicit models
use the solver's joint drive. Explicit models can therefore require different stability tuning.
When an explicit-actuator simulation diverges, check mass and inertia, target scaling and update
rate, effort limits, gains, time step, and solver convergence settings. Tune armature from the
motor and transmission model alongside those parameters. See the `OmniPhysics articulation
stability guide
<https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html>`_
for the solver-side background.

.. figure:: ../../_static/actuators/armature-clip.webp
    :align: center
    :width: 100%
    :alt: Five pendulums with increasing armature responding to the same command.

.. figure:: ../../_static/actuators/armature-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Position step response for an armature sweep.

.. figure:: ../../_static/actuators/armature-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Position step response for an armature sweep.


Friction
^^^^^^^^

Joint friction resists motion independently of the PD command. On a joint spun free (no stiffness or
damping), higher friction bleeds off velocity faster, so the pendulum coasts to rest sooner. Isaac
Lab exposes static (:attr:`~isaaclab.actuators.ActuatorBaseCfg.friction`), dynamic
(:attr:`~isaaclab.actuators.ActuatorBaseCfg.dynamic_friction`), and viscous
(:attr:`~isaaclab.actuators.ActuatorBaseCfg.viscous_friction`) friction. Use it to model gearbox
stiction and drag rather than to stabilize a controller.

.. note::

    Friction conventions depend on the backend. PhysX uses dimensionless static and dynamic
    coefficients in Isaac Sim 4.5, and effort values [N or N·m, depending on joint type] in 5.0
    and later. OVPhysX uses dimensionless static and dynamic coefficients. Newton uses a dry-friction
    effort and has no separate dynamic-friction value. All three use viscous damping
    [N·s/m or N·m·s/rad, depending on joint type].

.. figure:: ../../_static/actuators/friction-clip.webp
    :align: center
    :width: 100%
    :alt: Five free-spinning pendulums with increasing joint friction decaying at different rates.

.. figure:: ../../_static/actuators/friction-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Joint-velocity decay for a friction sweep.

.. figure:: ../../_static/actuators/friction-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Joint-velocity decay for a friction sweep.


Effort limit
^^^^^^^^^^^^

The effort limit is the torque ceiling the motor can produce [N·m or N]. The clip below drives an
:class:`~isaaclab.actuators.IdealPDActuator` swing-up from hanging to horizontal against a
~2.94 N·m gravity-hold torque, with the model's ``effort_limit`` swept over
:math:`[1, 2, 3, 4, 6]` N·m.

This curve is the clearest "saturation shapes the transient, gravity sets the steady state"
artifact. Limits comfortably above the ~2.94 N·m hold torque reach the horizontal target -- faster
with more headroom -- while limits below it never get there: the joint sags and settles where the
gravity torque (:math:`\approx 2.94 \sin\theta`) matches the ceiling. Along the way a saturated PD
oscillates: while the demand exceeds the limit, the applied torque pins at the ceiling and the
damping term is entirely clipped away, leaving undamped, constant-torque behavior until the demand
falls back inside the clip range. The takeaway when tuning: an effort limit below the load's static
demand does not merely slow the joint, it removes the controller's ability to damp itself.

.. figure:: ../../_static/actuators/effort-limit-clip.webp
    :align: center
    :width: 100%
    :alt: Five pendulums with increasing effort limit holding or failing against gravity.

.. figure:: ../../_static/actuators/effort-limit-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Applied joint torque for an effort-limit sweep.

.. figure:: ../../_static/actuators/effort-limit-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Applied joint torque for an effort-limit sweep.


Velocity limit
^^^^^^^^^^^^^^

The velocity limit is the no-load speed of a :class:`~isaaclab.actuators.DCMotor` [rad/s or m/s]:
the achievable torque decreases linearly as the joint approaches it, defining the four-quadrant
torque-speed envelope. The curve below is that envelope for a range of velocity limits -- a lower
limit shrinks the usable speed band and clamps torque earlier. A DC motor consumes ``velocity_limit``
for this torque-speed clipping. An implicit group exposes it through the soft velocity-limit view,
while ``joint_velocity_limit`` requests a solver constraint. Whether that constraint is enforced depends on the
backend, as described in :ref:`newton-velocity-limits`.

.. figure:: ../../_static/actuators/velocity-limit-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Torque-speed envelope for a velocity-limit sweep.

.. figure:: ../../_static/actuators/velocity-limit-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Torque-speed envelope for a velocity-limit sweep.


Command delay
^^^^^^^^^^^^^

A :class:`~isaaclab.actuators.DelayedPDActuator` lags every command by a fixed number of physics
steps drawn from ``[min_delay, max_delay]`` at reset. The clip below sweeps the delay over
:math:`[0, 6, 12, 24, 48]` physics steps (0--133 ms at :math:`dt = 1/360\text{ s}`) against a
square-wave position command: the more delayed pendulums visibly trail the reference. Randomizing
the delay between resets is a common domain-randomization technique for closing the sim-to-real gap
on real transport lag.

.. figure:: ../../_static/actuators/delay-clip.webp
    :align: center
    :width: 100%
    :alt: Five pendulums with increasing command delay trailing the same square-wave command.

.. figure:: ../../_static/actuators/delay-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Command-versus-response timeline for a delay sweep.

.. figure:: ../../_static/actuators/delay-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Command-versus-response timeline for a delay sweep.


Implicit vs. explicit
^^^^^^^^^^^^^^^^^^^^^

In this single-joint demo at :math:`dt = 1/360\text{ s}`, an implicit actuator and an ideal-PD
explicit actuator with identical gains produce nearly the same response. This is not a general
equivalence or stability guarantee: the implicit path uses the solver's joint drive while the
explicit model evaluates the PD law once per step. The overlaid curve below shows the two responses
for the same stiffness and damping. This is why a policy trained on implicit actuators may not
transfer unchanged to explicit ones -- and why the explicit joint's ``data.joint_stiffness`` /
``data.joint_damping`` read zero, since those gains now live in the model.

.. figure:: ../../_static/actuators/implicit-vs-explicit-curve-light.png
    :class: only-light
    :align: center
    :width: 80%
    :alt: Overlaid implicit and explicit PD step responses at identical gains.

.. figure:: ../../_static/actuators/implicit-vs-explicit-curve-dark.png
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Overlaid implicit and explicit PD step responses at identical gains.


.. _actuators-runtime-api:

Runtime API: ``articulation.actuators``
---------------------------------------

Group access
^^^^^^^^^^^^

:attr:`~isaaclab.assets.Articulation.actuators` is an
:class:`~isaaclab.actuators.ActuatorCollection`, a read-only ``Mapping`` from group name to actuator
model. Membership is fixed after construction, so you can look up and iterate groups but not add,
replace, or delete them:

.. code-block:: python

    legs = robot.actuators["legs"]          # the ActuatorBase for the "legs" group
    for name, actuator in robot.actuators.items():
        print(name, type(actuator).__name__)

Configure topology before creating the articulation:

.. code-block:: python

    robot_cfg.actuators["gripper"] = ImplicitActuatorCfg(...)
    robot = Articulation(robot_cfg)

At runtime, both ``robot.actuators["gripper"] = ...`` and
``del robot.actuators["gripper"]`` raise :class:`TypeError`. Membership and joint coverage are
construction-time invariants.

Logical groups and execution batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Named entries such as ``hips`` and ``knees`` retain separate configuration and access identities.
Isaac Lab may combine disjoint compatible stateless groups for execution without changing the
group tensors returned by ``robot.actuators["hips"]``. Each joint can belong to at most one group;
construction raises :class:`ValueError` when resolved joint selections overlap. Groups remain
separate when they use incompatible classes, are stateful or neural-network models, or run through
a native controller.
Set commands and perform lifecycle operations through the articulation and its
:class:`~isaaclab.actuators.ActuatorCollection`; execution batching is private.

Commands, telemetry, and lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Setting actuator commands.** The mutable ``command`` view contains the desired position,
velocity, and effort received by the actuator models. Use its index setters for the common case
(environment/joint index lists or tensors) and its mask setters when you already hold boolean Warp
masks. All are keyword-only:

.. code-block:: python

    import torch

    ids = torch.tensor([0, 1], device=robot.device)          # first two joints
    env_ids = torch.arange(robot.num_instances, device=robot.device)
    sub = torch.zeros((env_ids.numel(), ids.numel()), device=robot.device)
    robot.actuators.command.set_position_index(value=sub, joint_ids=ids, env_ids=env_ids)

Each index selector must contain unique environment and joint indices. Duplicate indices launch
concurrent writes to the same destination, so write ordering and the resulting value are undefined.
Deduplicate the selectors or use a boolean mask; a mask selects each destination at most once.

By default ``value`` is shaped ``(len(env_ids), len(joint_ids))``. Pass ``full_data=True`` when
``value`` is already a full ``(num_instances, num_joints)`` command buffer, so you don't have to
build a per-index sub-tensor: the same scatter kernel runs either way, but the source is then read
at full-buffer coordinates.

**Reading commands and telemetry.** ``command`` exposes the desired commands staged for actuator
processing. For Isaac Lab-managed models, ``joint_command`` exposes the processed commands produced
for the simulated joints. Native paths bypass this view, so it is not submitted-command telemetry
there. Newton-native controllers process inside the solver, while the PhysX native adapter processes
during ``write_data_to_sim()``. Read the underlying arrays through ``.torch`` (or ``.warp``):

.. code-block:: python

    desired_position = robot.actuators.command.position.torch
    processed_effort = robot.actuators.joint_command.effort.torch  # Isaac Lab-managed path
    applied = robot.actuators.applied_torque.torch      # after clipping [N·m or N]
    computed = robot.actuators.computed_torque.torch   # before clipping [N·m or N]

:attr:`~isaaclab.actuators.ActuatorCollection.computed_torque` is the model output before clipping
and :attr:`~isaaclab.actuators.ActuatorCollection.applied_torque` is the value after clipping (for
implicit actuators these are the approximate torques the model records for reward/telemetry use).

**Randomizing gains.** Runtime gain updates depend on the actuator implementation. Managed
environments should use the ``randomize_actuator_gains`` event, which updates actuator-owned values,
implicit solver drives, and native-controller parameters through the appropriate path. There is no
generic collection-level stiffness or damping writer.

**Lifecycle.** You do not call ``compute()`` or ``submit_commands()`` yourself. The articulation
runs ``actuators.reset()`` on environment resets. Each call to
:meth:`~isaaclab.assets.Articulation.write_data_to_sim` invokes actuator compute, staging, and
submission; call it before advancing the simulation, or rely on the scene or environment loop.

.. _actuators-migrating-setters:

.. rubric:: Migrating from the deprecated setters

Joint commands used to be set on the articulation itself. Those methods are deprecated forwarders to
the collection and emit a :class:`DeprecationWarning`:

.. code-block:: python

    # Before (deprecated)
    robot.set_joint_position_target(target, joint_ids=ids)

    # After
    robot.actuators.command.set_position_index(value=target, joint_ids=ids)

LEAPP-exported action terms must keep using the annotated articulation ``*_index`` or ``*_mask``
setters until the exporter supports collection setters. Other runtime code should migrate to the
collection API.

The old data reads move too: ``articulation.data.joint_pos_target`` becomes
``robot.actuators.command.position``, and ``data.computed_torque`` / ``data.applied_torque`` become
``robot.actuators.computed_torque`` / ``robot.actuators.applied_torque``. See the
:doc:`Isaac Lab 3.0 migration guide <../../migration/migrating_to_isaaclab_3-0>` for the full table.


Value resolution: USD vs. ActuatorCfg
-------------------------------------

Every gain and limit can come from either the USD joint-drive prim or the actuator config. The rule
is precedence by *specification*: a config field left as ``None`` inherits the value authored on the
USD prim, while a field you set on the config overrides it. Actuator parameters come from the joint
drive prims and the config -- not from the robot schema, which only supplies joint/body ordering
conventions.

To see exactly which value won for each joint, set
:attr:`~isaaclab.assets.ArticulationCfg.actuator_value_resolution_debug_print` to ``True`` on the
articulation config; the collection logs a table of USD value, config value, and applied value for
every joint whose sources disagree or whose config field was left unspecified (shown as
``Not Specified``). For configuration examples and the compact limit table, see
:ref:`how-to-write-articulation-config`.


.. _actuators-native:

Newton native actuators
------------------------

By default Isaac Lab runs explicit actuator models in Python, once per step, outside the solver
(typically on the selected Torch or Warp device). This Isaac Lab execution path is deprecated.
Set :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` to ``True`` to instead run the
explicit models **inside the Newton solver**:

.. code-block:: python

    from isaaclab.sim import SimulationCfg

    sim_cfg = SimulationCfg(use_newton_actuators=True)

**What changes.** With the flag on, each explicit actuator config is translated into a
``NewtonActuator`` USD prim and stepped by the physics engine rather than by
:meth:`ActuatorCollection.compute`. On the Newton backend the actuators run inside the
CUDA-graph-captured region. Implicit actuators are unaffected: their gains are written to the
solver and PD runs there as before, so implicit joints keep working exactly the same. The PhysX
backend can also consume these Newton-authored actuators through its adapter, so the authoring is
shared across backends. On CUDA, PhysX attempts to capture graphable Newton actuator staging, model
execution, and telemetry publication. Capture failures fall back to eager execution. Stateful Newton
actuators cannot be nested inside a caller-owned CUDA graph; let the PhysX adapter manage them instead.

Newton owns a separate native execution aggregation path. When native actuator handling is active,
the native controller remains the execution owner. Isaac Lab retains the named logical groups as the
configuration and access view, and stages their commands and telemetry.

**Supported models.** The authoring maps each supported config to a set of USD schemas:

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Config
      - Newton schemas
    * - :class:`~isaaclab.actuators.IdealPDActuatorCfg`
      - ``NewtonPDControlAPI`` + ``NewtonMaxEffortClampingAPI``
    * - :class:`~isaaclab.actuators.DCMotorCfg`
      - ``NewtonPDControlAPI`` + ``NewtonDCMotorClampingAPI``
    * - :class:`~isaaclab.actuators.DelayedPDActuatorCfg`
      - ideal PD + ``NewtonActuatorDelayAPI``
    * - :class:`~isaaclab.actuators.RemotizedPDActuatorCfg`
      - delayed PD + ``NewtonPositionBasedClampingAPI``
    * - :class:`~isaaclab.actuators.ActuatorNetMLPCfg` /
        :class:`~isaaclab.actuators.ActuatorNetLSTMCfg`
      - ``NewtonNeuralControlAPI`` (+ ``NewtonDCMotorClampingAPI``)

**USD-authored actuators survive.** The Lab config takes precedence per joint: for every joint
covered by an explicit Lab actuator config, any existing ``NewtonActuator`` prim on that joint is
replaced by one synthesized from the config. Joints **not** covered by any Lab config keep their
USD-authored ``NewtonActuator`` prims untouched -- so you can hand-author actuators on a subset of
joints and let the config drive the rest.

.. warning::

    With ``use_newton_actuators=True``, every explicit actuator config must be supported. An
    unsupported config raises before native authoring. Disable ``use_newton_actuators`` to use the
    Isaac Lab execution path, or select a supported config.

.. note::

    Under native actuators the delay is a **fixed** lag: the schema authors only ``max_delay``, so
    ``min_delay`` is dropped and a :class:`~isaaclab.actuators.DelayedPDActuator` does not randomize
    its delay between resets the way it does on the Isaac Lab path. [#native_delay]_

.. [#native_delay] This is a known asymmetry between the host and native delay paths; the fix is
   tracked as a separate code change outside this documentation.


Backend notes and non-goals
---------------------------

The submit stage differs per physics backend, but the collection interface above is identical on all
of them:

.. list-table::
    :header-rows: 1
    :widths: 24 76

    * - Backend
      - Submit behavior
    * - PhysX
      - The Isaac Lab-managed path pushes processed position, velocity, and effort staging buffers
        through the PhysX Tensor API. The native adapter processes commands during
        ``write_data_to_sim()`` and submits raw position and velocity targets plus its computed
        effort. A fused reorder gather runs first when a non-identity joint ordering is active.
    * - OVPhysX
      - The post-clip ``applied_torque`` is pushed as the effort together with the raw position and
        velocity target buffers (not the processed staging buffers) via OV ``set_attribute`` tensor
        bindings; every raw setter write is also eagerly mirrored into the binding at set time. A
        fused reorder gather runs first when a non-identity joint ordering is active.
    * - Newton / MJWarp
      - Targets are written into the solver's bound control arrays (effort feed-forward, plus
        ``joint_act`` under native actuators); the solver's built-in joint drive runs PD for
        implicit joints and adds the effort buffer as feed-forward.

For explicit actuators on any backend, remember that the solver's stiffness and damping for those
joints are zero -- the model owns the gains.

This page does **not** cover:

* Motion generators or low-level control modes -- see :doc:`motion_generators`.
* Cross-backend policy transfer and solver-dynamics differences -- see
  :doc:`physical-backends/sim-to-sim-policy-transfer`.

.. seealso::

    * :ref:`how-to-write-articulation-config` -- authoring configs and the full gain/limit value
      resolution walkthrough.
    * :ref:`import-new-asset-ensure-drives-exist` -- making sure joint drives exist on an imported
      asset so actuators can bind.
    * ``Joints actuate in PhysX but not in a Newton-based backend`` in
      :doc:`../../refs/troubleshooting` -- the USD drive / backend actuation pitfall.
    * :mod:`isaaclab.actuators` -- the full actuator model and config API reference.
