Fixed
^^^^^

* Fixed ``distance_to_camera`` incorrectly mapping to ``DistanceToImagePlaneSD`` in the OVRTX
  renderer backend. It now correctly uses ``DistanceToCameraSD``, matching the intended semantics
  of eye-space ray length versus perpendicular image-plane distance.
