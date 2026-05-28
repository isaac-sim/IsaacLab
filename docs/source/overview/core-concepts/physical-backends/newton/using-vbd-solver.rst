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
rigid-body scenes usually use
:class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` or
:class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg` so one
solver advances rigid bodies and VBD advances deformable particles.

Start from a Supported Deformable Task
--------------------------------------

Before adding VBD to a new task, first run one of the experimental Franka
deformable tasks:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Soft-Franka-v0 --num_envs 1 --visualizer kit

For the surface-deformable cloth variant, use:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Lift-Cloth-Franka-v0 --num_envs 1 --visualizer kit

Both tasks configure MJWarp for the rigid Franka and VBD for the deformable
object through
:class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`.
Use these tasks as starting points for asset setup, solver coupling, and contact
tuning.

Add a VBD Physics Preset
------------------------

Tasks that support multiple physics options usually store ``SimulationCfg.physics``
as a :class:`~isaaclab_tasks.utils.hydra.PresetCfg`. For deformable Newton tasks,
the preset can use a small :class:`~isaaclab_newton.physics.NewtonCfg` subclass
to carry :class:`~isaaclab_contrib.deformable.NewtonModelCfg` alongside the
normal Newton fields:

.. code-block:: python

    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_contrib.deformable import NewtonModelCfg


    @configclass
    class DeformableNewtonCfg(NewtonCfg):
        model_cfg: NewtonModelCfg | None = None


The Franka soft-body task defines a ``newton_mjwarp_vbd`` preset that couples
MJWarp and VBD:

.. literalinclude:: ../../../../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_franka_soft/franka_soft_env_cfg.py
    :language: python
    :start-at: class PhysicsCfg
    :end-at: default = newton_mjwarp_vbd
    :emphasize-lines: 4-32

The important pieces are:

* Add a Newton physics preset whose value is ``DeformableNewtonCfg``.
* Use :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` when rigid
  bodies and deformables must interact in the same scene.
* Use ``soft_solver_cfg=VBDSolverCfg(integrate_with_external_rigid_solver=True)``
  inside a coupled solver so VBD advances only the deformable particles.
* Add :class:`~isaaclab_contrib.deformable.NewtonModelCfg` when body-particle or
  self-contact values need task-level tuning.
* Keep the preset at the same config path used by the task's
  :class:`~isaaclab.sim.SimulationCfg`, for example ``env.sim.physics``.

You can select the deformable Newton preset globally:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Lift-Soft-Franka-v0 presets=newton_mjwarp_vbd

or select the physics field directly:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Lift-Soft-Franka-v0 env.sim.physics=newton_mjwarp_vbd

Use the direct path override when only one task field should use the VBD preset.
Use ``presets=newton_mjwarp_vbd`` when you want every matching preset field in
the task config to resolve to that preset. Isaac Lab training scripts accept
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
    * - ``shape_material_ke``
      - Default: ``None`` [N/m]. Optional override for all rigid collision-shape contact stiffness values in the Newton model. Use this when the rigid-side material parsed from the asset is not appropriate for deformable contact.
    * - ``shape_material_kd``
      - Default: ``None`` [N*s/m]. Optional override for all rigid collision-shape contact damping values in the Newton model.
    * - ``shape_material_mu``
      - Default: ``None``. Optional override for all rigid collision-shape friction values in the Newton model. Body-particle friction depends on both the soft contact and rigid shape friction coefficients.


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
      - Use ``coupling_mode="two_way"``, then increase ``soft_contact_mu`` and rigid-side ``shape_material_mu``. Also check gripper actuator stiffness and effort limits.
    * - The deformable barely deforms.
      - Reduce material stiffness, ``soft_contact_ke``, or shape contact stiffness.
    * - Contact chatters or bounces.
      - Increase ``soft_contact_kd`` or material damping, and consider using more substeps.
    * - Cloth passes through itself.
      - Enable ``particle_enable_self_contact``, increase ``particle_self_contact_radius`` if the active self-contact thickness is too small, increase ``particle_self_contact_margin`` if contacts are missed, and use a positive ``particle_collision_detection_interval``.
    * - Self-contact is too expensive.
      - Increase ``particle_collision_detection_interval``, reduce mesh resolution, or disable self-contact until the rest of the scene is tuned.

For implementation details of the VBD and coupled solver managers, see
:doc:`newton-manager-abstraction`.
