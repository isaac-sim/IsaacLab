Changed
^^^^^^^

* Removed per-step host synchronizations from the Lift progress rewards.
* Disabled point-cloud markers in the Lift observation config. Drawing every point each step was
  costly and, with RTX rendering, put the markers into camera images. Set
  ``observations.perception.object_point_cloud.params["visualize"] = True`` to draw them.
* Stored Cartpole direct joint indices on the device, removing host-to-device index uploads and CUDA
  synchronizations from its observations, rewards, terminations, resets, and effort writes.
* Removed redundant action copies in the Cartpole and Reorient direct tasks.
