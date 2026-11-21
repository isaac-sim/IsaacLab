* Labeled contact-sensor pose, force, and contact-point element axes with the resolved sensor
  order, including history and filtered-force layouts. Preserved known component axes when
  body or target-frame names were unavailable.
* Kept shared PVA storage compatible with LEAPP 0.8.1 by inheriting quantity annotations from
  its base data interface.
* Kept OSC task-space gains internal to exported effort computation instead of labeling them
  as joint gains.
