* Changed Kit ``--xr`` camera panels for the fixed-base G1 task: it shows head and wrist panels,
  and head-panel uploads are capped at 20 Hz (wrist panels at 10 Hz).
* Changed fixed-base G1 to re-render three times after each reset (``num_rerenders_on_reset = 3``), so
  its head and wrist images are not stale after a reset. This adds render work per reset when cameras
  are rendered.
