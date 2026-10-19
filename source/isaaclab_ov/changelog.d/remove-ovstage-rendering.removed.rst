* **Breaking:** Removed OVStage rendering, physics/rendering stage sharing, and the
  ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` and ``ISAAC_LAB_SHARE_OVSTAGE`` switches. OVRTX now
  always clones into its native scene; OVPhysX retains its required private physics stage.
  The public ``isaaclab_ov.stage`` resource was removed; OVPhysX owns its stage internally.
  Downstream renderers should use ``OvrtxReplicateContext`` and OVRTX native bindings.
