* Fixed the Shadow Hand and Allegro Hand reorientation task videos missing the hands in play mode;
  they now record a close view of one hand and its cube.
* Fixed the reorientation goal cube displaying behind the hand, outside the recording view. It now
  follows the hand at ``goal_marker_offset`` from the in-hand position; the camera tasks keep their
  fixed position inside the tiled camera view.
