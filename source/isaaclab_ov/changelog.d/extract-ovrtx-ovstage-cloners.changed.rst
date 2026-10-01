* Moved clone-plan interpretation out of OVRTX camera initialization and into its clone context. Both scene
  paths consumed the same prepared copies after camera overrides were authored. The renderer cloned and
  exported only assets routed to OVRTX, including spawned and shared assets.
