Changed
^^^^^^^

* Removed per-step host synchronizations from the Lift progress rewards.
* Disabled point-cloud markers in the Lift observation config. Drawing every point each step was
  costly and, with RTX rendering, put the markers into camera images. Set
  ``observations.perception.object_point_cloud.params["visualize"] = True`` to draw them.
