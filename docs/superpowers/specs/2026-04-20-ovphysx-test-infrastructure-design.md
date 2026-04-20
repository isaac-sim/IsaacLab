# OVPhysX Test Infrastructure — Design Spec

**Issue:** Needs creation — \[OVPHYSX\] Test infrastructure
**Date:** 2026-04-20
**Status:** Draft

## Summary

Establish comprehensive test coverage for the OVPhysX backend by:
1. Adding OVPhysX as a parametrized backend in core `isaaclab/test/` interface tests
2. Copying PhysX backend-specific tests into `isaaclab_ovphysx/test/` with setup adaptations
3. Extending the mock infrastructure to support all OVPhysX asset and sensor types

This is a cross-cutting concern that touches every other OVPhysX spec.

## Interface Tests (core isaaclab)

### Files to Modify

| Test File | What to Add |
|-----------|-------------|
| `test/assets/test_articulation_iface.py` | Already has OVPhysX (PR #4852) — verify it's complete |
| `test/assets/test_rigid_object_iface.py` | Add `"ovphysx"` backend parametrization |
| `test/assets/test_rigid_object_collection_iface.py` | Add `"ovphysx"` backend parametrization |

### Pattern to Follow

From the existing `test_articulation_iface.py` OVPhysX integration:
1. Add `"ovphysx"` to `@pytest.mark.parametrize("backend", [...])` 
2. Create mock bindings in the test fixture that match the OVPhysX backend
3. Mock bindings must produce data with correct shapes and dtypes
4. Test fixture creates the asset using the mock, not a real ovphysx instance

### No Sensor Interface Tests Currently

There are no `test_*_iface.py` files for sensors. If they are created in the future, add OVPhysX parametrization following the same pattern.

## Backend-Specific Tests (isaaclab_ovphysx)

### Files to Create (copy from PhysX)

| Source (isaaclab_physx) | Target (isaaclab_ovphysx) | Changes |
|-------------------------|---------------------------|---------|
| `test/assets/test_rigid_object.py` | `test/assets/test_rigid_object.py` | Swap PhysxCfg → OvPhysxCfg |
| `test/assets/test_rigid_object_collection.py` | `test/assets/test_rigid_object_collection.py` | Swap config |
| `test/assets/test_deformable_object.py` | `test/assets/test_deformable_object.py` | Swap config (Phase 2) |
| `test/sensors/test_contact_sensor.py` | `test/sensors/test_contact_sensor.py` | Swap config |
| `test/sensors/test_imu.py` | `test/sensors/test_imu.py` | Swap config |
| `test/sensors/test_pva.py` | `test/sensors/test_pva.py` | Swap config |
| `test/sensors/test_frame_transformer.py` | `test/sensors/test_frame_transformer.py` | Swap config |

### Allowed Modifications When Copying

Per the guiding principle — only setup modifications:
- Replace `PhysxCfg` / `SimulationCfg` with `OvPhysxCfg`
- Replace Kit-dependent USD asset paths with local test assets (if assets require Nucleus)
- Skip tests that require Kit-specific APIs (AppLauncher, live viewport, Fabric)
- Add `@pytest.mark.skipif(not ovphysx_available)` guards
- Adjust GPU buffer sizes in OvPhysxCfg if defaults differ

### NOT Allowed When Copying

- Do not change test logic or assertions
- Do not change expected values or tolerances (if physics differs, that's an ovphysx bug for @marcodiiga)
- Do not skip tests because they fail — mark them `xfail` with a reason and file a bug

## Mock Infrastructure

### Current State

`isaaclab_ovphysx/test/mock_interfaces/views/mock_ovphysx_bindings.py` provides:
- `MockTensorBinding` — simulates ovphysx TensorBinding with numpy backend
- `MockOvPhysxBindingSet` — factory for creating matched bindings

### Extensions Needed

| Mock | Purpose | For |
|------|---------|-----|
| `MockRigidBodyBindingSet` | Single rigid body (num_bodies=1) | RigidObject tests |
| `MockRigidBodyCollectionBindingSet` | Multiple rigid bodies | RigidObjectCollection tests |
| `MockDeformableBodyBindingSet` | Nodal state for deformable | DeformableObject tests |

Each mock must:
- Support `read(tensor)` and `write(tensor, indices, mask)` operations
- Return correct shapes and dtypes for the tensor type
- Support `set_random_data()` for test initialization

## Test Asset Management

Some PhysX tests reference USD assets from Nucleus (`{ISAAC_NUCLEUS_DIR}/...`). OVPhysX is kit-less and may not have Nucleus access.

**Solution:** Use local test USD assets that are checked into the repo:
- Simple articulation USD (e.g., cartpole, single joint)
- Simple rigid body USD (e.g., cube, sphere)
- Check these into `source/isaaclab_ovphysx/test/data/` or reference existing test data

PR #4852 already moved some test USD data into the repo — extend this pattern.

## CI Integration

### Test Commands

```bash
# Run all OVPhysX interface tests
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/ -k ovphysx

# Run all OVPhysX backend tests
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/

# Run specific backend test
./isaaclab.sh -p -m pytest source/isaaclab_ovphysx/test/assets/test_rigid_object.py
```

### CI Requirements

From PR #4852 discussion:
- Interface tests (mock-based) can run on any CI runner without GPU
- Backend-specific tests require the ovphysx wheel installed
- GPU integration tests require GPU CI runners

## Dependencies

- Each test file depends on its corresponding implementation being complete
- Mock infrastructure should be extended incrementally as each component is implemented
- Interface test changes can be done early (mock-only, no ovphysx wheel needed)

## Estimated Scope

- Interface test modifications: ~100 lines total (add parametrization to 3 files)
- Backend test copies: ~1500 lines total (7 test files with config swaps)
- Mock extensions: ~200 lines
- Test asset management: ~50 lines of config
