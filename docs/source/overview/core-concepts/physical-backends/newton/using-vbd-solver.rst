.. _newton-using-vbd:

VBD Solver
==========

Vertex Block Descent (VBD) is a Newton solver for cloth and soft-body
simulation. In Isaac Lab, VBD is enabled by selecting a
:class:`~isaaclab_newton.physics.NewtonCfg` whose ``solver_cfg`` is provided by
:mod:`isaaclab_contrib.deformable`.

VBD support is experimental. The solver managers, configuration fields, and
recommended tuning values may change while Newton deformable support is under
active development. A task that works with PhysX or with Newton's MuJoCo-Warp
solver may still need deformable assets, materials, contacts, and coupling tuned
before it works well with VBD.

VBD is usually exposed through a task-specific physics preset rather than a
general ``newton_vbd`` preset. Deformable-only scenes can use
:class:`~isaaclab_contrib.deformable.VBDSolverCfg` directly. Robot or
rigid-body scenes usually use one of the coupled configs so one solver advances
rigid bodies and VBD advances deformable particles:

* :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` — alternates
  the rigid (MJWarp) and VBD substeps. Use it when the same robot should both
  contact and feel the deformable.
* :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` —
  partitions the model among named entries and exposes selected bodies from one
  entry as *proxies* in another entry's view via lagged impulses (see
  :ref:`newton-vbd-proxy-coupling` below). Use it when only a few rigid bodies
  (e.g. a gripper) need to interact with the deformable.
* :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg` —
  alternates Featherstone and VBD; supports kinematic one-way coupling.

Start from a Supported Deformable Task
--------------------------------------

Before adding VBD to a new task, first run one of the experimental Franka
deformable tasks:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

For the surface-deformable cloth variant, use:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Cloth-Franka --num_envs 1 --visualizer kit

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Cloth-Franka --num_envs 1 --visualizer kit

Both tasks configure MJWarp for the rigid Franka and VBD for the deformable
object through
:class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`.
Use these tasks as starting points for asset setup, solver coupling, and contact
tuning.

Add a VBD Physics Preset
------------------------

Tasks that support multiple physics options usually store ``SimulationCfg.physics``
as a :class:`~isaaclab_tasks.utils.hydra.PresetCfg`. For deformable Newton tasks,
the preset is a plain :class:`~isaaclab_newton.physics.NewtonCfg` whose solver
config carries :class:`~isaaclab_contrib.deformable.NewtonModelCfg` through its
:class:`~isaaclab_contrib.deformable.NewtonModelSolverCfg` base class.

The Franka soft-body task defines a ``newton_mjwarp_vbd`` preset that couples
MJWarp and VBD:

.. literalinclude:: ../../../../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py
    :language: python
    :start-at: class PhysicsCfg
    :end-before: newton_mjwarp_vbd_proxy: NewtonCfg

The important pieces are:

* Add a Newton physics preset whose value is a
  :class:`~isaaclab_newton.physics.NewtonCfg`.
* Use :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` when rigid
  bodies and deformables must interact in the same scene.
* Use ``soft_solver_cfg=VBDSolverCfg(integrate_with_external_rigid_solver=True)``
  inside a coupled solver so VBD advances only the deformable particles.
* Set the solver config's ``model_cfg`` to a
  :class:`~isaaclab_contrib.deformable.NewtonModelCfg` when body-particle or
  self-contact values need task-level tuning.
* Keep the preset at the same config path used by the task's
  :class:`~isaaclab.sim.SimulationCfg`, for example ``env.sim.physics``.

You can select the deformable Newton preset globally:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka physics=newton_mjwarp_vbd

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka physics=newton_mjwarp_vbd

or select the physics field directly:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka env.sim.physics=newton_mjwarp_vbd

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka env.sim.physics=newton_mjwarp_vbd

Use the direct path override when only one task field should use the VBD preset.
Use ``physics=newton_mjwarp_vbd`` when you want every matching preset field in
the task config to resolve to that preset. Isaac Lab training commands accept
these Hydra overrides after the regular command line flags; no separator is
needed for the examples above.


Check Task and Asset Compatibility
----------------------------------

VBD uses the Newton model built from the task assets. When adding VBD to a new
task, validate the following before tuning solver parameters:

* The task must already be compatible with the Newton backend. If a rigid-only
  ``newton_mjwarp`` preset fails during model construction, fix the asset or task
  configuration first.
* The scene must include Newton-compatible deformable assets and materials. Use
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg`
  for volume deformables and
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableBodyMaterialCfg`
  for cloth or surface deformables.
* Coupled robot tasks should start with ``coupling_mode="two_way"`` when the
  robot should feel contact forces from the deformable object.
* Contact-heavy scenes usually need task-specific ``num_substeps``,
  :class:`~isaaclab_contrib.deformable.VBDSolverCfg`, and
  :class:`~isaaclab_contrib.deformable.NewtonModelCfg` values. Start from the
  Franka soft-body or cloth preset that most closely resembles the scene.
* Use a small visual smoke test before training. Confirm that the deformable
  spawns, renders, deforms, and contacts rigid bodies as expected.

VBD Solver Parameters
---------------------

The following fields are specific to
:class:`~isaaclab_contrib.deformable.VBDSolverCfg`. They are grouped by the part
of the solver they affect.

Core Solve
^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``iterations``
      - Default: ``10``. Number of VBD iterations per substep. Increasing this value improves deformation and contact convergence, especially for stiff materials or rigid gripper contacts, but increases runtime.
    * - ``integrate_with_external_rigid_solver``
      - Default: ``False``. Set to ``True`` when VBD is used inside a coupled solver so the rigid sub-solver owns rigid-body integration. Leave ``False`` for deformable-only VBD scenes.


Self-Contact
^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``particle_enable_self_contact``
      - Default: ``False``. Enables deformable self-contact. Use this for cloth folds or soft bodies that collide with themselves. It increases contact work and usually needs additional tuning.
    * - ``particle_self_contact_radius``
      - Default: ``0.005`` [m]. Effective self-contact thickness. VBD applies vertex-triangle and edge-edge self-contact response when the current primitive distance is smaller than this radius.
    * - ``particle_self_contact_margin``
      - Default: ``0.005`` [m]. Self-contact candidate search distance. VBD uses this envelope when building self-contact lists, then applies contact response using ``particle_self_contact_radius``. Keep this greater than or equal to the radius to avoid missed contacts.
    * - ``particle_collision_detection_interval``
      - Default: ``-1``. Controls how often self-contact detection runs. A negative value detects before initialization only. ``0`` detects before and immediately after initialization. A positive value ``k`` detects before every ``k`` VBD iterations.
    * - ``particle_vertex_contact_buffer_size``
      - Default: ``32``. Preallocation size for each vertex's vertex-triangle self-contact buffer. Increase it if dense folds or high-resolution cloth exceed the default capacity.
    * - ``particle_edge_contact_buffer_size``
      - Default: ``64``. Preallocation size for each edge's edge-edge self-contact buffer. Increase it if dense folds or high-resolution cloth exceed the default capacity.
    * - ``particle_topological_contact_filter_threshold``
      - Default: ``2``. Filters contacts between mesh primitives that are close in topology. Increase this to suppress contact between neighboring elements of the same surface. Values greater than ``3`` can significantly increase compute time.
    * - ``particle_rest_shape_contact_exclusion_radius``
      - Default: ``0.0`` [m]. Filters self-contact candidates whose rest-configuration distance is shorter than this distance. Increase it when rest-neighbor contacts produce unwanted resistance.


Coupled Solver Parameters
-------------------------

Use the coupled solver configs when one solver should advance rigid bodies and
VBD should advance deformables:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``rigid_solver_cfg``
      - Rigid-body sub-solver configuration. :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` uses :class:`~isaaclab_newton.physics.MJWarpSolverCfg`; :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg` uses :class:`~isaaclab_newton.physics.FeatherstoneSolverCfg`.
    * - ``soft_solver_cfg``
      - VBD sub-solver configuration. In coupled scenes, set ``integrate_with_external_rigid_solver=True`` so VBD advances only deformable particles.
    * - ``coupling_mode="one_way"``
      - Rigid solver advances first, and VBD reacts to the updated rigid poses. The rigid solver does not feel particle contact forces.
    * - ``coupling_mode="two_way"``
      - Contact reactions from deformables are injected into the rigid solver before the rigid step, then VBD advances deformables against the shared contacts. Use this for manipulation tasks where the robot should be pushed back by deformable contact.
    * - ``coupling_mode="kinematic"``
      - Available on :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg`. Rigid bodies are kinematically updated by Featherstone, and VBD reacts to them. The rigid solver does not feel particle contacts.

The rigid solver parameters still matter. For example, MJWarp's ``nconmax`` and
``njmax`` must be large enough for the rigid contacts in the scene, and
``ccd_iterations`` can affect fast rigid contacts near deformables. See
:doc:`mjwarp-solver` for the MJWarp-side parameters.


.. _newton-vbd-proxy-coupling:

Proxy-Coupled MJWarp + VBD
--------------------------

:class:`~isaaclab_contrib.coupling.CouplerProxyCfg` is an alternative
MJWarp + VBD coupling that wraps Newton's
:class:`newton.solvers.experimental.coupled.SolverCoupledProxy`. Instead of
alternating two full-model substeps, the model is **partitioned** between a
set of named solver entries. Each directed proxy mapping names a source entry
(rigid, e.g. MJWarp) and a destination entry (soft, e.g. VBD), then exposes
selected source bodies to the destination solver as *proxies* — virtual copies
that the destination collides against. Contact feedback is returned to the
source solver as lagged impulses. This typically scales better than the
alternating coupling when only a small set of rigid bodies (e.g. the fingers of
a gripper) actually needs to touch the deformable, since the bulk of the
articulation is solved purely by MJWarp without seeing the particle contacts.

Choose between the two MJWarp + VBD presets based on how much of the rigid
model needs to interact with the deformable:

.. list-table:: Alternating MJWarp + VBD versus proxy coupling
    :header-rows: 1
    :widths: 20 40 40

    * - Consideration
      - Alternating ``newton_mjwarp_vbd``
      - Proxy ``newton_mjwarp_vbd_proxy``
    * - Interaction model
      - Runs MJWarp and VBD over the shared model and injects deformable
        reactions into the rigid solve in two-way mode.
      - Partitions the model into solver views and exposes only selected source
        bodies or particles as virtual proxies in the destination view.
    * - Advantages
      - Provides direct, same-substep two-way feedback for contacts across the
        rigid model. It is the simpler choice when many robot links may contact
        the deformable.
      - Restricts coupled contact work to a small interface, which can scale
        better when only a gripper or another small body subset interacts with
        the deformable. Named entries also allow supported solver combinations
        beyond the dedicated MJWarp + VBD manager.
    * - Trade-offs and limits
      - Uses a dedicated MJWarp + VBD path and performs shared contact work even
        when only a few rigid bodies need deformable contact.
      - Feedback is lagged or staggered and can be more timestep-sensitive.
        Newton's proxy solver currently supports at most two entries, does not
        support joints that cross entry boundaries, and couples only explicitly
        selected proxy bodies or particles.
    * - Choose it when
      - Tight two-way feedback across much of the articulation matters more than
        limiting the coupling interface.
      - Contact is localized to a known body subset and the reduced interface is
        worth the proxy approximation and topology restrictions.

The Franka soft-body task ships a ``newton_mjwarp_vbd_proxy`` preset (the new
default for ``Isaac-Lift-Soft-Franka``) that demonstrates the typical
configuration:

.. literalinclude:: ../../../../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py
    :language: python
    :start-at: newton_mjwarp_vbd_proxy: NewtonCfg
    :end-before: physx: PhysxCfg = PhysxCfg()
    :dedent: 4

What the selectors do:

* ``entries`` contains one
  :class:`~isaaclab_contrib.coupling.CouplerEntryCfg` per sub-solver. Each
  entry has a stable ``name``, its own ``solver_cfg``, and explicit model
  ownership selectors.
* An entry's ``bodies`` selectors are full Newton body-label regex strings.
  By default, joints inherit their child body's owner and
  shapes inherit their body's owner. Use ``all_particles=True`` to own all
  deformable particles and ``include_static_shapes=True`` to own world
  geometry. Bodies, particles, joints, and shapes may be assigned to at most
  one entry; unassigned elements remain outside the nested solver views.
* ``proxies`` contains directed
  :class:`~isaaclab_contrib.coupling.CouplerProxyMappingCfg` mappings. Each mapping
  names its ``source`` and ``destination`` entries, then uses ``bodies`` to
  select the source bodies that the destination solver should collide against.
  Only bodies that own at least one ``newton.ShapeFlags.COLLIDE_SHAPES`` shape
  are kept.
* In the snippet above, the entire Franka articulation is routed to MJWarp,
  while the deformable particles and static table/world shapes are routed to
  VBD. Only the ``panda_hand`` and the two fingers are exposed as proxies —
  so VBD only ever sees three rigid proxies regardless of how many links the
  arm has.

.. important::

    The coupler currently rejects
    :class:`~isaaclab_newton.physics.KaminoSolverCfg` entries and
    :class:`~isaaclab_newton.physics.MPMSolverCfg` entries configured with
    ``project_outside_colliders=True``, as well as
    :class:`~isaaclab_newton.physics.MJWarpSolverCfg` entries configured with
    ``use_mujoco_cpu=True``. These configurations require manager-specific
    build, forward-kinematics, reset, or per-step lifecycle hooks that are not
    yet available to nested solvers. Use MPM with
    ``project_outside_colliders=False`` and GPU MJWarp, or run these solvers
    through their standalone managers until nested lifecycle support is added.

Key proxy-specific parameters:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``CouplerProxyMappingCfg.mode``
      - Default: ``"lagged"``. ``"lagged"`` syncs source begin poses and end
        velocities, then rewinds lagged feedback before the destination solve.
        ``"staggered"`` syncs source end poses and end velocities directly.
        Lagged is the safer default; staggered can be tighter but is more
        sensitive to timestep.
    * - ``CouplerProxyCfg.iterations``
      - Default: ``1``. Number of relaxation iterations per coupled substep.
        Increase it when proxy contact feedback needs more accuracy.
    * - ``CouplerProxyMappingCfg.collide_interval``
      - Default: ``None`` (every proxy pass). How often the proxy collision
        pipeline rebuilds candidate pairs. Increase it for cheaper but slightly
        staler proxy contacts.
    * - ``CouplerProxyMappingCfg.mass_scale``
      - Default: ``1.0``. Multiplier for the virtual inertia of proxy bodies
        in the VBD view. Increase it to make proxies behave more like fixed
        obstacles to VBD.

Body selectors must use full Newton body-label regexes, such as
``/World/envs/env_.*/Robot``. Proxy mappings also accept raw Newton body ids.

Try the demo:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          # zero-agent visual smoke test (default preset is now the proxy-coupled one)
          uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

          # scripted pick-and-lift via state machine
          uv run python scripts/environments/state_machine/lift_franka_soft.py --num_envs 1

          # explicitly select the alternating-substep preset instead
          uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 presets=newton_mjwarp_vbd


   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          # zero-agent visual smoke test (default preset is now the proxy-coupled one)
          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

          # scripted pick-and-lift via state machine
          ./isaaclab.sh -p scripts/environments/state_machine/lift_franka_soft.py --num_envs 1

          # explicitly select the alternating-substep preset instead
          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 presets=newton_mjwarp_vbd


Contact and Material Parameters
-------------------------------

Contact Model
^^^^^^^^^^^^^

:class:`~isaaclab_contrib.deformable.NewtonModelCfg` applies contact parameters
to the finalized Newton model:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``soft_contact_ke``
      - Default: ``1.0e3`` [N/m]. Stiffness for body-particle and particle self-contact. Increase it to reduce clipping through rigid shapes or through other deformable particles. If it is too high, the object can stop visibly deforming or require more VBD iterations and substeps.
    * - ``soft_contact_kd``
      - Default: ``1.0e-2`` [N*s/m]. Contact damping. Increase it to reduce chatter or bouncing. Too much damping can make contact response sticky or overdamped.
    * - ``soft_contact_mu``
      - Default: ``0.5``. Friction coefficient for body-particle and particle self-contact. Increase it when a gripper cannot carry the deformable object without slipping.

To set rigid collision-shape contact properties (``ke``, ``kd``, ``mu``) for
shapes that lack an explicit per-asset material, use
:class:`~isaaclab_newton.physics.NewtonShapeCfg` on ``NewtonCfg.default_shape_cfg``
instead. Per-asset materials override these defaults.


Volume Deformable Materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use
:class:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg`
for volume deformables:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``density``
      - Default: ``1.0`` [kg/m^3]. Material density. Higher density increases particle mass and inertia, so the object accelerates and deforms less for the same contact forces.
    * - ``particle_radius``
      - Default: ``0.008`` [m]. Particle contact radius used by Newton. Increase it when contacts are missed or detected too late. If it is too large relative to the mesh resolution, contacts can start too early.
    * - ``k_mu``
      - Default: ``1.0e5`` [Pa]. First Lame material parameter. Higher values make the deformable object stiffer and usually require more VBD iterations, more substeps, or a smaller timestep.
    * - ``k_lambda``
      - Default: ``1.0e5`` [Pa]. Second Lame material parameter. Higher values make the deformable object stiffer and usually require more VBD iterations, more substeps, or a smaller timestep.
    * - ``k_damp``
      - Default: ``0.0`` [Pa*s]. Damping for tetrahedral elements. Increase it to reduce oscillations after deformation, but avoid overdamping if the object should rebound.


Surface Deformable Materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use
:class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableBodyMaterialCfg`
for cloth or surface deformables:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``density``
      - Default: ``1.0`` [kg/m^3]. Material density. Higher density increases particle mass and inertia.
    * - ``particle_radius``
      - Default: ``0.008`` [m]. Particle contact radius used by Newton.
    * - ``tri_ke``
      - Default: ``1.0e4`` [Pa]. Triangle area-preserving stiffness. Increase it to reduce cloth stretch.
    * - ``tri_ka``
      - Default: ``1.0e4`` [Pa]. Triangle area stiffness. Increase it to reduce cloth area change.
    * - ``tri_kd``
      - Default: ``1.5e-6`` [Pa*s]. Triangle area damping. Increase it to reduce cloth vibration after stretching.
    * - ``edge_ke``
      - Default: ``5.0`` [N*m]. Bending stiffness. Increase it for stiffer cloth folds; decrease it for softer draping.
    * - ``edge_kd``
      - Default: ``1.0e-2`` [N*m*s]. Bending damping. Increase it to damp fold oscillations.

Tuning Workflow
---------------

Use the following sequence when bringing up a new VBD task:

1. Run one of the supported Franka deformable tasks and confirm your
   installation, visualizer, and deformable rendering path work.
2. Add a task-specific VBD or coupled VBD preset copied from the closest
   supported task.
3. Run a small visual smoke test with ``--num_envs 1`` before training.
4. Tune deformable material stiffness and damping until the object deforms in
   the expected range without rigid contact.
5. Increase ``num_substeps`` or decrease ``dt`` if the object is unstable before
   increasing stiffness further.
6. Increase :attr:`~isaaclab_contrib.deformable.VBDSolverCfg.iterations` when
   contacts or stiff materials do not converge within a substep.
7. Tune :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_ke` to
   reduce rigid/deformable clipping, then tune
   :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_mu` for grip
   and :attr:`~isaaclab_contrib.deformable.NewtonModelCfg.soft_contact_kd` for
   chatter.
8. Enable self-contact only after body-particle contact is stable, then tune
   ``particle_self_contact_radius`` for active self-contact thickness,
   ``particle_self_contact_margin`` for missed contacts, and
   ``particle_collision_detection_interval`` for detection frequency.
9. Increase ``num_envs`` and profile only after the single-environment scene is
   stable.


Symptoms and First Parameters to Check
--------------------------------------

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Symptom
      - First parameters to check
    * - Rigid bodies visibly clip through the deformable.
      - Increase ``soft_contact_ke``, VBD ``iterations``, ``num_substeps``, or the deformable material ``particle_radius``.
    * - The robot cannot lift the deformable.
      - Use ``coupling_mode="two_way"``, then increase ``soft_contact_mu`` and the rigid-side shape friction (per-asset material ``mu`` or ``NewtonShapeCfg.mu``). Also check gripper actuator stiffness and effort limits.
    * - The deformable barely deforms.
      - Reduce material stiffness, ``soft_contact_ke``, or shape contact stiffness.
    * - Contact chatters or bounces.
      - Increase ``soft_contact_kd`` or material damping, and consider using more substeps.
    * - Cloth passes through itself.
      - Enable ``particle_enable_self_contact``, increase ``particle_self_contact_radius`` if the active self-contact thickness is too small, increase ``particle_self_contact_margin`` if contacts are missed, and use a positive ``particle_collision_detection_interval``.
    * - Self-contact is too expensive.
      - Increase ``particle_collision_detection_interval``, reduce mesh resolution, or disable self-contact until the rest of the scene is tuned.

For implementation details of the VBD managers and Newton coupler, see
:doc:`newton-manager-abstraction`.
