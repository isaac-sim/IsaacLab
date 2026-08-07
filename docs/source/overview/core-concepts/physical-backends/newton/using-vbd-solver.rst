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
rigid-body scenes can use either:

* :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` for the proxy coupling
  used by the core Franka tasks.
* :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` for the
  opt-in shared-model example with custom substep ordering.

Start from a Supported Deformable Task
--------------------------------------

.. note::

   The ``Isaac-Lift-Soft-Franka`` task requires automatic tetrahedralization.
   Install its optional dependencies before running the examples below:

   .. code-block:: bash

      uv sync --inexact --extra tetrahedralization

   With the legacy installer:

   .. code-block:: bash

      ./isaaclab.sh -i tetrahedralization

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
:class:`~isaaclab_contrib.coupling.CouplerProxyCfg`.
Use these tasks as starting points for asset setup, solver coupling, and contact
tuning.

Add a VBD Physics Preset
------------------------

Tasks that support multiple physics options usually store ``SimulationCfg.physics``
as a :class:`~isaaclab_tasks.utils.hydra.PresetCfg`. For deformable Newton tasks,
the preset is a plain :class:`~isaaclab_newton.physics.NewtonCfg` whose solver
config carries :class:`~isaaclab_contrib.deformable.NewtonModelCfg` through its
:class:`~isaaclab_contrib.deformable.NewtonModelSolverCfg` base class.

The Franka soft-body and cloth tasks define task-specific proxy presets.

The important pieces are:

* Add a Newton physics preset whose value is a
  :class:`~isaaclab_newton.physics.NewtonCfg`.
* Use :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` with named
  :class:`~isaaclab_contrib.coupling.CouplerEntryCfg` entries to partition the
  rigid bodies and deformable particles between MJWarp and VBD.
* Add :class:`~isaaclab_contrib.coupling.CouplerProxyMappingCfg` entries for the
  collidable rigid bodies exposed to VBD. Leave
  ``integrate_with_external_rigid_solver=False`` for proxy-coupled VBD entries.
* Set the solver config's ``model_cfg`` to a
  :class:`~isaaclab_contrib.deformable.NewtonModelCfg` when body-particle or
  self-contact values need task-level tuning.
* Keep the preset at the same config path used by the task's
  :class:`~isaaclab.sim.SimulationCfg`, for example ``env.sim.physics``.

You can select the deformable Newton preset globally:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka physics=newton_mjwarp_vbd_proxy

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka physics=newton_mjwarp_vbd_proxy

or select the physics field directly:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka env.sim.physics=newton_mjwarp_vbd_proxy

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Lift-Soft-Franka env.sim.physics=newton_mjwarp_vbd_proxy

Use the direct path override when only one task field should use the VBD preset.
Use ``physics=newton_mjwarp_vbd_proxy`` when you want every matching preset
field in the task config to resolve to that preset. Isaac Lab training commands
accept these Hydra overrides after the regular command line flags; no separator is
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
* Proxy-coupled robot tasks should expose only the collidable bodies needed for
  deformable contact.
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
      - Default: ``False``. Set to ``True`` only when a manual manager integrates rigid bodies in the shared model. Proxy-coupled entries use partitioned model views and leave this ``False``.


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


Custom MJWarp + VBD Parameters
------------------------------

The opt-in
:class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg` runs
MJWarp and VBD over one shared model. Import
:mod:`isaaclab_contrib.custom_coupling.tasks` explicitly before using its
registered task.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``rigid_solver_cfg``
      - MJWarp configuration for rigid bodies.
    * - ``soft_solver_cfg``
      - VBD configuration. Set ``integrate_with_external_rigid_solver=True``
        so VBD advances only particles.
    * - ``coupling_mode="one_way"``
      - Advance rigid bodies first, then particles without rigid reaction
        forces.
    * - ``coupling_mode="two_way"``
      - Inject particle reactions before MJWarp, then advance VBD with the same
        contacts.

MJWarp ``nconmax`` and ``njmax`` must cover the rigid contacts and constraints
in the scene. ``ccd_iterations`` can affect fast rigid contacts near
deformables. See :doc:`mjwarp-solver` for the rigid-solver parameters.

Use the custom manager for direct shared-model substep ordering. Use proxy
coupling when deformable contact is localized to selected rigid bodies.


.. _newton-vbd-proxy-coupling:

Proxy-Coupled MJWarp + VBD
--------------------------

:class:`~isaaclab_contrib.coupling.CouplerProxyCfg` is the coupling used by
the core Franka tasks. It partitions the model between named solver entries and
exposes selected source bodies to the destination solver as proxies. Contact
feedback returns to the source solver as lagged impulses. This keeps deformable
contact work localized to the rigid bodies that need it, such as a gripper hand
and fingers. The pinned proxy solver supports at most two entries and rejects
joints that cross entry boundaries.

The core Franka soft-body task demonstrates the proxy configuration:

.. literalinclude:: ../../../../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_soft_env_cfg.py
    :language: python
    :start-at: newton_mjwarp_vbd_proxy: NewtonCfg
    :end-before: isaacsim_physx: PhysxCfg = PhysxCfg()
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
  VBD. Only the ``panda_hand`` and the two fingers are exposed as proxies,
  so VBD sees three rigid proxies regardless of the number of arm links.

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

          # zero-agent visual smoke test
          uv run python scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

          # scripted pick-and-lift via state machine
          uv run python scripts/environments/state_machine/lift_franka_soft.py --num_envs 1

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          # zero-agent visual smoke test
          ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka --num_envs 1 --visualizer kit

          # scripted pick-and-lift via state machine
          ./isaaclab.sh -p scripts/environments/state_machine/lift_franka_soft.py --num_envs 1


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
2. Add a task-specific VBD or proxy-coupled VBD preset copied from the closest
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
      - Check that the gripper bodies are included in the proxy, then increase ``soft_contact_mu`` and rigid-side shape friction (per-asset material ``mu`` or ``NewtonShapeCfg.mu``). Also check gripper actuator stiffness and effort limits.
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
