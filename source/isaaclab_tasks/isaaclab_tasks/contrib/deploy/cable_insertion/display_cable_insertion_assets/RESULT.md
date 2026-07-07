# 2584N111 DisplayPort Cord Plug SimReady Result

## Final artifact

- USD: `2584n111_displayport_cord_plug_latch_removed_simready.usd`
- Preview: `thumbnail.png`
- Validation reports:
  - `minimum-usd.json`
  - `asset-validator.json`
  - `simready-profile.json`
  - `ovrtx-render-service.json`

## Provenance

- Source STEP SHA-256: `6b56128cdf97bbb220f05bd4d159525873ac663042d90512f490208491421c17`
- Skill Hub clone: `NVIDIA/skills`
- Skill Hub commit: `8b6938719346266262d21a16c5ab20c1ea8156f9`
- Remote run root: `/home/horde/nvidia-skills-cad2simready-runs/20260529-2584n111-plug-latch-removed`
- SimReady profile: `Prop-Robotics-Neutral@1.0.0`

## Validation

- Minimum USD: PASS
- Omni Asset Validator: PASS, with warnings only
- Geometry validation: PASS, with warnings only
- Physics validation: PASS
- SimReady profile validation: PASS
- OVRTX render service: PASS

## Notes

- Final USD has `metersPerUnit = 1.0`, `upAxis = Z`, 20 mesh prims, 24 material bindings, 4 colliders, and 4 rigid bodies.
- The Material Agent completed, but only 1 of 4 material VLM predictions succeeded; visual material accuracy should be treated as limited.
- The Physics Agent completed, but its semantic identification called the part an automotive diagnostic adapter. The final USD still passes the selected SimReady validation profile.
- Remaining validator messages are warnings: repeated indexed primvars and 11 non-manifold vertices on one mesh.
