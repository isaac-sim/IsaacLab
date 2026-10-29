* Fixed the Warp velocity and reach reward terms reading the command buffer of the first environment
  created in the process instead of their own environment's.
* Fixed the Warp ``terrain_out_of_bounds`` termination using the terrain bounds of the first environment
  created in the process.
