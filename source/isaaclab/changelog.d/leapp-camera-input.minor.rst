Added
^^^^^

* Exposed camera RGB data as a typed ``state/camera/image`` LEAPP input compatible with
  Isaac ROS Deploy.
* Preserved articulation joint names on LEAPP command-target inputs so deployment
  runtimes can reorder controller commands safely.
* Added an action-term contract for simulation writes that are owned by the deployment
  controller and therefore omitted from policy outputs.
