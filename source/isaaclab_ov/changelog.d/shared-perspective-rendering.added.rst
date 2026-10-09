* Added renderer-owned perspective cameras through an empty ``CameraRenderSpec.camera_prim_paths``.
  Shared scene preparation and native resources with sensor products while allowing independent
  resize, product settings, and cleanup. Perspective cameras saw all environment partitions;
  sensor cameras retained per-environment isolation.
* **Breaking:** Removed implicit ambient illumination from OVRTX products. Scenes without authored lights remained unlit;
  add scene lights to illuminate their geometry.
* **Breaking:** Removed implicit RGB output and unsupported-output skipping in OVRTX product authoring.
  Request supported outputs explicitly through ``CameraCfg.data_types``; invalid requests now raised ``ValueError``.
