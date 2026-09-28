Added
^^^^^

* Added a ``valid`` mask argument to :meth:`~isaaclab_tasks.utils.success_monitor.SuccessMonitor.success_update`
  so callers can skip outcomes without filtering them first.

Changed
^^^^^^^

* Removed per-step host synchronizations from the Lift progress rewards.
* Disabled point-cloud markers in the Lift observation config. Drawing every point each step was
  costly and, with RTX rendering, put the markers into camera images. Set
  ``observations.perception.object_point_cloud.params["visualize"] = True`` to draw them.
* Stored Cartpole direct joint indices on the device, removing host-to-device index uploads and CUDA
  synchronizations from its observations, rewards, terminations, resets, and effort writes.
* Removed redundant action copies in the Cartpole and Reorient direct tasks.
* Made :class:`~isaaclab_tasks.utils.success_monitor.SuccessMonitor` updates free of host synchronizations,
  removing about 8.5 synchronizations per step from the Lift camera task.
* Changed :meth:`~isaaclab_tasks.utils.success_monitor.SuccessMonitor.get_mean_success_rate` to return a 0-d
  tensor instead of a Python float, so per-step logging does not synchronize. Call ``.item()`` where a float
  is needed.
* Stored Pendulum MARL joint indices on the device, removing host-to-device index uploads and CUDA
  synchronizations from its terminations, resets, and effort writes.
* Logged the Cartpole direct and Pendulum MARL success rates as device tensors instead of synchronizing
  with ``.item()`` on every reset.
