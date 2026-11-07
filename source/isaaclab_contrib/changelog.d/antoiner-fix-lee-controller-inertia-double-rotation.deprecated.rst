* Deprecated the ``body_com_quat_b`` argument of
  ``isaaclab_contrib.utils.math.aggregate_inertia_about_robot_com``. It is now ignored and emits a
  :class:`DeprecationWarning`, since the link-frame inertias are rotated by the link orientation only.
  Pass ``None`` instead.
