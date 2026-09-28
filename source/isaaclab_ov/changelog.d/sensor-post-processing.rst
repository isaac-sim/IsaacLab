Changed
^^^^^^^

* Removed PPISP execution from OVRTX while preserving HDR buffer bindings for ``mdp.processed_image``
  observation terms. Added ``rgb_radiance`` with PPISP's existing camera-wide exposure setup;
  requesting both ``rgb_hdr`` and ``rgb_radiance`` shared the active HDR source. Existing
  ``CameraCfg.isp_cfg`` configurations remained supported through a compatibility adapter.
  HDR transfers between devices reused persistent storage.
