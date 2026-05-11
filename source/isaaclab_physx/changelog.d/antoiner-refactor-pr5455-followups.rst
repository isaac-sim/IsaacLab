Changed
^^^^^^^

* Reworded the FF-routing comments in
  :class:`~isaaclab_physx.assets.Articulation` to refer to "actuated DOFs"
  rather than splitting on implicit vs explicit, since the
  ``synch_torque_and_apply_implicit_feedforwards`` kernel operates on the full
  actuated DOF set.
