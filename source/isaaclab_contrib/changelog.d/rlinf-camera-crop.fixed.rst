* Kept visual augmentation in task observation processors and deterministic GPU image resizing in the RLinf adapter.
* Validated GR00T N1.5 train/eval image sizes on the driver before worker startup, preserving checkpoint metadata when adapting input dimensions.
* Resolved worker split settings once for observation preprocessing and padded hold actions.
