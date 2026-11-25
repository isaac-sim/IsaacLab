* Added manager-based ``IsaacContrib-Cartpole-Lidar`` and direct ``IsaacContrib-Cartpole-Lidar-Direct``
  tasks in :mod:`isaaclab_tasks.contrib.cartpole_lidar`. The policy reads two stacked lidar range
  images through the camera CNN. The critic still reads the joint state. Tiled capture requires the
  ``ovrtx`` extra.
