Fixed
^^^^^

* Fixed :class:`~isaaclab.sim.converters.UrdfConverter` and :class:`~isaaclab.sim.converters.MjcfConverter`
  producing a floating-base PhysX articulation with ``fix_base=True`` when a fixed joint attaches the root link to
  the world: the ``root_joint`` that the URDF importer adds, or the weld of an MJCF root body without joints. The
  Isaac Sim importers left the articulation root on the root link, which UsdPhysics treats as a floating base, so
  PhysX held the base only through that joint: ``is_fixed_base`` was False, root-pose writes were undone within a
  few steps, and joints near the base could gain energy. The converters rooted the articulation at that joint
  instead. Newton already treated these assets as fixed-base.
* Fixed ``fix_root_link=True`` raising ``NotImplementedError`` for an articulation rooted at its fixed world joint
  when the joint sets both bodies, as for the converted assets above. It enabled that joint instead, so these assets
  no longer got a second world joint, which Newton failed to import.

Changed
^^^^^^^

* Changed the articulations of these converted assets to fixed-base on PhysX. ``num_base_dofs`` is 0, the Jacobian,
  mass matrix and gravity compensation forces have no base degrees of freedom, a ``body_ordering`` must keep the root
  link first, and root-pose writes and ``init_state`` place the base. On Newton, an explicit
  ``articulation_root_prim_path`` has to name the fixed joint instead of the root link. USD files converted earlier,
  including those in a reused ``usd_dir``, keep the floating-base layout until they are converted again.
