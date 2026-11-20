* Removed the unused side RGB camera from ``presets=lidar`` and set that scene's default to 64
  environments. ``--num_envs 512`` still selects the larger count.
* Renamed the manager range-image term to ``LidarRangeImage`` and removed the unused flat point
  observation.
* Pointed the direct SKRL lidar value network at the 4-D joint state.
