Changelog
---------

1.1.0 (2026-08-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added VBD cable support.


1.0.0 (2026-08-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the opt-in :mod:`isaaclab_contrib.custom_coupling` example. Import
  :mod:`isaaclab_contrib.custom_coupling.tasks` explicitly to register
  ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``.

Deprecated
^^^^^^^^^^

* Deprecated :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`. Use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` for MJWarp and VBD
  coupling, or :class:`~isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg`
  to stay on the manual coupler.

Removed
^^^^^^^

* **Breaking:** Moved ``NewtonCoupledMJWarpVBDManager`` and its reaction kernel out
  of :mod:`isaaclab_contrib.deformable` and into the opt-in
  :mod:`isaaclab_contrib.custom_coupling` example, and removed the
  ``isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager`` module. Import the
  manager from :mod:`isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager`
  instead. Configurations that reference the manager through
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` keep working and
  now resolve to the moved class.

* **Breaking:** Removed ``CoupledFeatherstoneVBDSolverCfg`` and
  ``NewtonCoupledFeatherstoneVBDManager`` from
  :mod:`isaaclab_contrib.deformable`. Switch the rigid solver to MJWarp and use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` or the opt-in
  :mod:`isaaclab_contrib.custom_coupling` example.


0.5.2 (2026-07-29)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the Lee controller base to read body masses and inertias from the
  public-order :attr:`~isaaclab.assets.ArticulationData.body_mass` and
  :attr:`~isaaclab.assets.ArticulationData.body_inertia` buffers instead of the
  backend-order tensor view, so per-body terms pair correctly with the
  public-order center-of-mass buffers under a non-identity
  :attr:`~isaaclab.assets.ArticulationCfg.body_ordering`.


0.5.1 (2026-07-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the type-only import of ``ModelView`` in
  :class:`~isaaclab_contrib.coupling.CouplerEntryCfg`, which Newton moved from ``newton`` to
  ``newton.solvers.experimental.coupled``.


0.5.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_contrib.coupling`, exposing
  :class:`~isaaclab_contrib.coupling.coupler.NewtonCouplerManager`
  together with the
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerCfg`
  base config and two algorithm-specific subclasses:
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerProxyCfg`
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`) and
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerAdmmCfg`
  (wrapping :class:`newton.solvers.experimental.coupled.SolverCoupledADMM`).
  The coupler partitions the Newton model among explicit, named
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerEntryCfg`
  entries, instantiates each sub-solver from its config, and connects entries
  through named proxy mappings or symmetric ADMM contact pairs.

* Added support for prim-path regex strings (e.g.
  ``"/World/envs/env_.*/MyCube"``) in the body-selector lists of
  :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerEntryCfg`
  and :class:`~isaaclab_contrib.coupling.coupler_cfg.CouplerProxyMappingCfg`.
  Raw Newton body ids may also be given directly as integers in
  :attr:`~isaaclab_contrib.coupling.coupler_cfg.CouplerProxyMappingCfg.bodies`.

* Added :class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelSolverCfg`,
  a shared solver-config base whose ``model_cfg``
  (:class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelCfg`) is
  applied to the finalized Newton model. The VBD and coupler configs inherit it.

* Added implicit MPM support for coupled-solver entries, including per-entry
  substeps and in-place stepping.

Changed
^^^^^^^

* **Breaking:** Removed ``ISAACLAB_CONTRIB_METADATA`` and ``ISAACLAB_CONTRIB_EXT_DIR`` from
  :mod:`isaaclab_contrib`. These constants were populated from the now-deleted
  ``config/extension.toml`` Kit extension manifest.
* Removed the model-global ``shape_material_ke/kd/mu`` fields from
  :class:`~isaaclab_contrib.deformable.newton_manager_cfg.NewtonModelCfg`, which
  filled every rigid shape's material and clobbered per-asset materials. Set
  per-shape defaults through
  :class:`~isaaclab_newton.physics.NewtonShapeCfg` on ``NewtonCfg.default_shape_cfg``
  instead; per-asset materials now override those defaults. The model-global
  ``soft_contact_ke/kd/mu`` fields are unchanged.
* Changed :class:`~isaaclab_contrib.deformable.DeformableObject` to follow the backend's
  default physics context (``isaaclab_newton.cloner.PHYSICS_CONTEXT``) directed by the asset
  cfg: USD clones now accompany Newton replication only under Kit, instead of unconditionally,
  matching the other Newton assets.

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``.

Fixed
^^^^^

* Fixed proxy-coupled source solvers configured for external contacts to receive
  contacts from Newton's shared collision pipeline. Proxy destinations continue
  to use their entry-local collision pipeline.
* Fixed :class:`~isaaclab_contrib.coupling.CouplerCfg` to reject inactive
  entries and unsupported nested solver configurations.

* Fixed proxy collision fallback and MPM contact and graph-capture policy.


0.4.7 (2026-07-13)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed simulation mesh discovery in :class:`~isaaclab_contrib.deformable.DeformableObject` to detect
  deformable sim API schemas authored as unregistered tokens (e.g. by Newton), so surface deformables
  no longer fall back to treating the visual mesh as the simulation mesh.


0.4.6 (2026-07-01)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed two-way rigid-deformable contact reactions with Newton shape margins.


0.4.5 (2026-06-08)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed Newton deformable clone replication and Fabric particle sync setup.


0.4.4 (2026-06-06)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed RLinf optional dependency installation on DGX Spark and aarch64 by
  replacing ``decord`` with ``decord2`` in the ``rlinf`` extras.


0.4.3 (2026-06-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed deformable Newton test presets to rely on iterative MuJoCo Warp line search.


0.4.2 (2026-06-04)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the TacSL visuotactile sensor demo, documentation, and tests to use
  current PhysX configuration and wrench APIs.


0.4.1 (2026-06-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``--rl_model_path`` CLI flag to ``play.py`` for evaluating RL-finetuned checkpoints.
  The base model architecture is loaded via ``--model_path`` and the RL-trained weights
  (``full_weights.pt``) are overlaid from the checkpoint directory.

Fixed
^^^^^

* Fixed Newton replicated-scene setup for deformable VBD managers to use
  clone-plan source prims.


0.4.0 (2026-05-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_contrib.deformable` with contributed Newton deformable
  asset and VBD solver support, including
  :class:`~isaaclab_contrib.deformable.DeformableObject`,
  :class:`~isaaclab_contrib.deformable.VBDSolverCfg`,
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`, and
  :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg` for
  one- and two-way rigid-deformable coupling.
* Added :class:`~isaaclab_contrib.deformable.NewtonModelCfg` for shared Newton
  deformable contact parameters.
* Added Newton deformable coupling documentation with Franka soft-body lift
  tuning guidance for
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` and
  :class:`~isaaclab_contrib.deformable.NewtonModelCfg`.

Fixed
^^^^^

* Fixed ``[rlinf]`` extra dependency declarations to avoid version conflicts with IsaacLab core
  (torch, transformers, tokenizers). Conflicting packages are now documented as manual ``--no-deps``
  installation steps.


0.3.2 (2026-05-12)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed ``_patched_reset`` monkey-patch in RLinf extension; use
  ``num_rerenders_on_reset`` env config instead.


0.3.1 (2026-05-09)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated TacSL visuotactile sensor camera configuration and examples to use
  :class:`~isaaclab.sensors.CameraCfg` and :class:`~isaaclab.sensors.Camera`
  instead of deprecated tiled-camera aliases.


0.3.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated multirotor asset and TacSL visuotactile sensor to wrap warp data
  property accesses with ``wp.to_torch()``.


0.2.1 (2026-02-03)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the multirotor asset to use the new base classes from the isaaclab_physx package.


0.2.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the multirotor asset to use the new base classes from the isaaclab_physx package.


0.1.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^


* Changed the quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion ordering.


0.0.2 (2026-01-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_contrib.sensors.tacsl_sensor` module with the TacSL tactile sensor implementation
  from :cite:t:`si2022taxim`.


0.0.1 (2025-12-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added initial implementation for multi rotor systems.
