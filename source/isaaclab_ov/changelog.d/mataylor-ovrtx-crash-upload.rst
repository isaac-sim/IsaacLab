Added
^^^^^

* Added Omniverse crash-report upload for the OVRTX renderer. Setting
  ``ISAACLAB_OVRTX_CRASH_UPLOAD`` configures the crash reporter before the
  renderer is constructed, so a crashing render process preserves and uploads its
  minidump. Requires ``ovrtx_extensions``, which is published on NVIDIA's internal
  index; the variable raises when it is set without that package installed, and
  leaving it unset keeps the previous behavior.
