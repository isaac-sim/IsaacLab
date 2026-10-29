* Added capability-aware shared PVA output storage and generated optional-output kernels.
* Construct and compile optional PVA updates on demand through cached factories. Using one feature subset or resetting a sensor does not construct or compile unused update variants.
* **Breaking:** Removed backend PVA data classes and ``BasePvaData``; import ``PvaData`` from ``isaaclab.sensors``. Pose and spatial components now share storage; initialize/reset/STOP use common valid defaults.
* Reused core IMU, frame-transformer, and sensor-reset operations while retaining backend-native acquisition.
