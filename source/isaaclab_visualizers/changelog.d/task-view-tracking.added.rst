* Added task views to the Newton GL visualizer: perspective views tied to an environment or an asset are
  rendered from the viewer camera, listed in **Camera View** as "Tracking Camera" beside the free "Perspective" view,
  and recordable without changing the window's view.
* Added editable Tracking Camera fields to the Newton GL **Camera View** section: environment, followed asset and
  body, eye, look-at, focal length, heading follow, and heading smoothing, with a reset to what the task declared. An
  unknown asset or ambiguous body is reported and the camera keeps its previous target.
