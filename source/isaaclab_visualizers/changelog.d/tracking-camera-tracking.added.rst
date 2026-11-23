* Added tracking camera support to the Newton GL visualizer. **Camera Source** lists it as "Tracking Camera" beside the
  free "Perspective" view, and a Tracking Camera group edits its environment, asset, body, eye, look-at, focal length
  and heading settings, with a reset to the task's declaration.
  "Reset to Task Default" also works while RTX is selected, and the window remembers the free Perspective camera's pose
  and focal length while another camera source is shown.
  The Tracking Camera's environment is always drawn, taking the last place of the Visible Envs count when it would
  otherwise be hidden, and "Show Only This Env" still draws only that environment.
