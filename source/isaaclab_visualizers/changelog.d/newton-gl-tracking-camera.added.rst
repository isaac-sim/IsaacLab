* Added the task's tracking camera to the Newton GL Camera Source list as "Tracking Camera", with an Environment field
  that picks the environment it follows. The free view is now called "Perspective" when it is the only one, and only
  the Playback section starts open.
* Added editable Tracking Camera fields to the Newton GL Camera View section: environment, followed asset and body,
  eye, look-at, focal length, heading follow, and heading smoothing, with a reset to what the task declared. An
  unknown asset or ambiguous body is reported and the camera keeps its previous target.
