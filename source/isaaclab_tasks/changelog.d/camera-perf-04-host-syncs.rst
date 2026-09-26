Changed
^^^^^^^

* Removed per-step host synchronizations from the Lift progress rewards and ``object_ee_distance``.
* Changed the default of ``visualize`` in ``isaaclab_tasks.core.lift.mdp.object_point_cloud_b`` to
  False. Drawing every point each step was costly and, with RTX rendering, put the markers into
  camera images. Pass ``visualize=True`` to draw them.
