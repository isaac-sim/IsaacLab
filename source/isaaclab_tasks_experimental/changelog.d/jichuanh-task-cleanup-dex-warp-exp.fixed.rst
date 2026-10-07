* Fixed the ``Isaac-Reorient-Cube-Allegro-Direct`` Warp environment measuring the object's
  orientation error from the ``(y, z, w)`` quaternion components instead of ``(x, y, z)``, which
  made its orientation reward, goal success, and goal resampling differ from the stable task.
* Fixed the ``Isaac-Reorient-Cube-Allegro-Direct`` Warp environment swapping the object's linear
  and angular velocity in its full observation.
