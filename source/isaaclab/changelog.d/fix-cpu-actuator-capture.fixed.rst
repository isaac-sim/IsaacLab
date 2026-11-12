* Fixed CPU graph capture of Torch actuator target commands and index selectors by passing zero-copy
  Warp views to the captured launches. Preserved eager execution and existing Warp array inputs.
